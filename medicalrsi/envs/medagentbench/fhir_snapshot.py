"""Snapshot of the MedAgentBench FHIR server (HAPI 7.6.0 in the medagentbench-fhir container) for the offline mock
(fhir_server.py). Pages through every resource type that holds data (Patient, Observation, Condition,
MedicationRequest, Procedure; every other declared type is empty, checked with _summary=count) with _count=5000 and
writes the full resources, in the server's own unfiltered search order and with every number's original text
(hapi_json.py), to <out>/<Type>.jsonl.gz, plus
<out>/manifest.json (counts, server version, time), <out>/metadata.json (the CapabilityStatement text, served
verbatim), <out>/search_params.json (declared search parameters and their types per resource type) and
<out>/patient_pid_order.json (Patient ids in HAPI's internal resource-PID order, from _sort=_pid: index scans return
rows in PID order; for the other types the numeric id is the PID).

    python -m medicalrsi.envs.medagentbench.fhir_snapshot --base http://localhost:8080/fhir/ --out data/medagentbench/fhir_snapshot
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
import urllib.request
from pathlib import Path

from medicalrsi.envs.medagentbench.hapi_json import dumps_compact, loads

TYPES = ("Patient", "Observation", "Condition", "MedicationRequest", "Procedure")


def get(url):
    with urllib.request.urlopen(url, timeout=600) as r:
        return loads(r.read().decode("utf-8"))           # decimals kept as Decimal: number text is preserved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8080/fhir/")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base = a.base.rstrip("/") + "/"
    meta = get(base + "metadata?_format=json")
    declared = [r["type"] for r in meta["rest"][0]["resource"]]
    nonempty = {}
    for t in declared:
        n = get(f"{base}{t}?_summary=count&_format=json").get("total", 0)
        if n:
            nonempty[t] = n
    assert set(nonempty) == set(TYPES), f"unexpected non-empty types: {nonempty}"
    with urllib.request.urlopen(base + "metadata?_format=json", timeout=600) as r:      # served verbatim by the mock
        (out / "metadata.json").write_text(r.read().decode("utf-8"))
    (out / "search_params.json").write_text(json.dumps(
        {r["type"]: {p["name"]: p.get("type") for p in r.get("searchParam", [])} for r in meta["rest"][0]["resource"]}))
    manifest = {"server": meta.get("software"), "fhirVersion": meta.get("fhirVersion"), "base": base,
                "taken": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "counts": {}}
    for t in TYPES:
        url, n, seen = f"{base}{t}?_count=5000&_format=json", 0, set()
        tmp = out / f".{t}.jsonl.gz.tmp"
        with gzip.open(tmp, "wt") as f:
            while url:
                b = get(url)
                for e in b.get("entry", []):
                    r = e["resource"]
                    key = r["id"]
                    if key in seen:
                        raise SystemExit(f"{t}: duplicate id {key} while paging")
                    seen.add(key)
                    f.write(dumps_compact(r) + "\n")
                    n += 1
                url = next((l["url"] for l in b.get("link", []) if l.get("relation") == "next"), None)
                if url and "_format=json" not in url:
                    url += "&_format=json"
        if n != nonempty[t]:
            raise SystemExit(f"{t}: paged {n} resources but _summary=count says {nonempty[t]}")
        tmp.replace(out / f"{t}.jsonl.gz")
        manifest["counts"][t] = n
        print(f"{t}: {n}", flush=True)
    b = get(base + "Patient?_sort=_pid&_count=5000&_format=json")
    pids = [e["resource"]["id"] for e in b.get("entry", [])]
    if len(pids) != manifest["counts"]["Patient"] or any(l.get("relation") == "next" for l in b.get("link", [])):
        raise SystemExit("Patient?_sort=_pid did not return every patient on one page")
    (out / "patient_pid_order.json").write_text(json.dumps(pids))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))


if __name__ == "__main__":
    main()
