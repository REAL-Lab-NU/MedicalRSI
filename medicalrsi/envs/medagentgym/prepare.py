"""Builds the MedAgentGym (EHR-SeqSQL) split from MedAgentGym's EHR-SeqSQL files and the EHRSQL MIMIC-III database,
both credentialed (PhysioNet).

  1. The answer of every sub-question is recomputed by running its gold SQL with the EHRSQL scorer's post-processing
     (fixed current time); items whose gold SQL fails or returns no rows are dropped.
  2. One item per distinct gold SQL with at most 10 answer rows; a workflow is one SQL structure (the gold query with
     every value masked). 26 training workflows are drawn among the structures with at least 30 items, with 20
     training and 10 test tasks each, and the order in which training rounds visit them.
  3. 6 unseen workflows (task39-task44): structures whose SQL pattern (table and column names masked as well) is less
     than 0.8 similar to every training pattern, with 10 test tasks each.
All draws use fixed seeds.

    python -m medicalrsi.envs.medagentgym.prepare --source <MedAgentGym>/ehr_seqsql --database <EHRSQL>/mimic_iii.sqlite \
        --out data/medagentgym
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import random
import re
from pathlib import Path

from medicalrsi.envs.medagentgym.sql import Database

BENCH = "EHRSQL"
N_TRAIN, N_TEST, N_IN, N_OUT = 20, 10, 26, 6
HELD_SHAPE_MAX = 0.8


def recompute(source: Path, db: Database) -> dict:
    """part -> items with the answers recomputed on the database."""
    out = {}
    for part in ("train", "test"):
        items, empty, failed = [], 0, 0
        for line in open(source / f"{part}_data.jsonl"):
            r = json.loads(line)
            rows, ok = db.run_gold(r["seqsql"])
            if not ok:
                failed += 1
                continue
            if not rows:
                empty += 1
                continue
            items.append(json.loads(json.dumps({
                "id": r["id"], "question": r["question"], "seed_question": r.get("seed_question"),
                "template": r["question_template"], "query": r["seqsql"], "answer": [list(x) for x in rows],
                "part": part})))
        out[part] = items
        print(f"{part}: {len(items)} items kept, {empty} with no rows and {failed} whose gold SQL failed dropped")
    return out


def skeleton(sql: str) -> str:
    """The SQL structure of a gold query: lower case, every quoted value and number masked."""
    q = re.sub(r"'[^']*'", "'S'", (sql or "").lower())
    q = re.sub(r"(?<![a-z_])\d+(\.\d+)?", "N", q)
    return re.sub(r"\s+", " ", q).strip()


def family(skel: str):
    """The tables a structure reads and the column it selects first."""
    tables = frozenset(re.findall(r"\b(?:from|join)\s+([a-z_]+)", skel))
    m = re.match(r"select\s+(?:distinct\s+)?(.+?)\s+from\b", skel)
    return tables, m.group(1) if m else skel[:40]


def shape(skel: str) -> str:
    """The SQL pattern of a structure: its table and column names masked as well."""
    s = re.sub(r"\b[a-z_0-9]+\.[a-z_0-9]+\b", "C", skel)
    s = re.sub(r"\b(from|join)\s+[a-z_0-9]+", r"\1 T", s)
    return re.sub(r"\s+", " ", s).strip()


def shape_sim(a: str, b: str) -> float:
    ta, tb = (set(re.findall(r"[a-z_]+|[<>=*()]+", x)) for x in (a, b))
    return len(ta & tb) / len(ta | tb)


def structures(source: Path, items: dict) -> dict:
    """structure -> items, one item per distinct gold SQL with at most 10 answer rows."""
    raw = {}
    for part in ("train", "test"):
        for line in open(source / f"{part}_data.jsonl"):
            r = json.loads(line)
            raw[r["id"]] = r
    by, seen = collections.defaultdict(list), set()
    for part in ("train", "test"):
        for x in items[part]:
            if "n_rows" in x or not x["answer"] or len(x["answer"]) > 10 or x["query"] in seen:
                continue
            seen.add(x["query"])
            by[skeleton(x["query"])].append(dict(x, seed_question=raw[x["id"]].get("seed_question"), part=part))
    return by


def _task(x: dict, sk: str, part: str, template) -> dict:
    return {"ehrsql_id": x["id"], "ehrsql_part": x["part"], "question": x["question"],
            "seed_question": x.get("seed_question"), "query": x["query"], "answer": x["answer"],
            "template": sk, "question_template": template, "part": part}


def first_split(by: dict, seed: int = 0):
    """26 training workflows (20 + 10 tasks each), their round order, and 6 unseen structures."""
    rng = random.Random(seed)
    elig = sorted(k for k, v in by.items() if len(v) >= N_TRAIN + N_TEST)
    ins = rng.sample(elig, N_IN)
    fam_in = {family(k) for k in ins}
    rest = [k for k in elig if k not in ins and family(k) not in fam_in]
    outs = rng.sample(rest, N_OUT)
    split, tasks, types = {"train": [], "test": [], "branch": {}}, {}, {}
    for k, sk in enumerate(ins + outs, 1):
        t = f"task{k}"
        held = k > N_IN
        pick = rng.sample(sorted(by[sk], key=lambda x: x["id"]), N_TEST if held else N_TRAIN + N_TEST)
        types[t] = {"template": sk, "heldout": held, "items": len(by[sk])}
        for i, x in enumerate(pick, 1):
            s = f"{t}_{i}"
            part = "test" if held or i > N_TRAIN else "train"
            split[part].append([BENCH, s])
            tasks[s] = _task(x, sk, part, x["template"])
    order = [f"task{k}" for k in range(1, N_IN + 1)]
    rng.shuffle(order)
    return split, tasks, {"types": types, "train_order": [f"{BENCH}/{t}" for t in order]}


def add_unseen(by: dict, sp: dict, ts: dict, ty: dict, seed, tag: str, replace: bool) -> None:
    """Six unseen workflows whose SQL pattern is less than HELD_SHAPE_MAX similar to every training pattern."""
    old_held = {t for t, v in ty["types"].items() if v["heldout"]}
    ind_shapes = {shape(v["template"]) for v in ty["types"].values() if not v["heldout"]}
    used = {v["template"] for v in ty["types"].values()}

    def sim(k):
        return max(shape_sim(shape(k), s) for s in ind_shapes)
    cand = sorted(k for k, xs in by.items() if len(xs) >= N_TEST and k not in used and sim(k) < HELD_SHAPE_MAX)
    rng = random.Random(seed)
    outs = rng.sample(cand, N_OUT)
    if replace:
        for t in old_held:
            del ty["types"][t]
        sp["test"] = [x for x in sp["test"] if x[1].split("_")[0] not in old_held]
        for s in [s for s in ts if s.split("_")[0] in old_held]:
            del ts[s]
        k0 = max(int(t[4:]) for t in old_held) + 1
    else:
        k0 = max(int(t[4:]) for t in ty["types"]) + 1
    for k, sk in enumerate(outs, k0):
        t = f"task{k}"
        ty["types"][t] = {"template": sk, "heldout": True, "items": len(by[sk]), "split": tag, "shape_sim": round(sim(sk), 3)}
        for i, x in enumerate(rng.sample(sorted(by[sk], key=lambda x: x["id"]), N_TEST), 1):
            s = f"{t}_{i}"
            sp["test"].append([BENCH, s])
            ts[s] = _task(x, sk, "test", x.get("template"))


def build(source: Path, db: Database):
    by = structures(source, recompute(source, db))
    sp, ts, ty = first_split(by)
    sp, ts, ty = copy.deepcopy(sp), copy.deepcopy(ts), copy.deepcopy(ty)
    add_unseen(by, sp, ts, ty, "v2b:0", "v2b", replace=True)
    add_unseen(by, sp, ts, ty, "v2c:0", "v2c", replace=False)
    drop = {t for t, v in ty["types"].items() if v.get("split") == "v2b"}
    for t in drop:
        del ty["types"][t]
    sp["test"] = [x for x in sp["test"] if x[1].split("_")[0] not in drop]
    ts = {s: v for s, v in ts.items() if s.split("_")[0] not in drop}
    return sp, ts, ty


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="MedAgentGym's ehr_seqsql folder (train_data.jsonl, test_data.jsonl)")
    ap.add_argument("--database", required=True, help="the EHRSQL MIMIC-III database (mimic_iii.sqlite)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sp, ts, ty = build(Path(a.source), Database(a.database))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    os.chmod(out, 0o700)
    (out / "split.json").write_text(json.dumps(sp))
    (out / "tasks.json").write_text(json.dumps(ts))
    (out / "types.json").write_text(json.dumps(ty, indent=1))
    held = sorted((t for t, v in ty["types"].items() if v["heldout"]), key=lambda t: int(t[4:]))
    print(f"{len(sp['train'])} training / {len(sp['test'])} test tasks; unseen workflows {held}")


if __name__ == "__main__":
    main()
