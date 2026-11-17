"""Process score for MedAgentBench v3, as for v1 and v2: query (the chart data the decision needs) and action (the
expected behaviour of refsol.ref_taskN, and "called FINISH"). All checks passing implies that refsol.taskN passes."""
from datetime import timedelta
from urllib.parse import parse_qs, urlparse

from . import refsol as R


def _role(m):
    return m.role if hasattr(m, "role") else m.get("role")


def _content(m):
    return (m.content if hasattr(m, "content") else m.get("content")) or ""


def _fetched(results, resource, mrn, codes=None):
    want = {c.lower() for c in (codes or [])}
    for m in results.history:
        c = _content(m).strip()
        if _role(m) != "agent" or not c.startswith("GET"):
            continue
        u = urlparse(c[3:].strip())
        if not u.path.rstrip("/").endswith("/" + resource):
            continue
        q = {k.lower(): ",".join(v) for k, v in parse_qs(u.query).items()}
        if mrn not in " ".join(q.get(k, "") for k in ("patient", "subject", "identifier", "_id")):
            continue
        got = {x.strip().lower() for x in q.get("code", "").split(",") if x.strip()}
        if want and got and not (got & want):
            continue
        return True
    return False


def _dedupe_checks(posts, ref, drug, dups, keep, label):
    stops = [pl for _, pl in posts if R.is_discontinuation(pl, ref, drug)]
    ids = [str(pl["id"]) for pl in stops if pl.get("id") is not None]
    return [(f"{label}: at least one discontinuation", bool(stops)),
            (f"{label}: one discontinuation per duplicate ({len(dups)})", len(stops) == len(dups)),
            (f"{label}: ids, when given, are duplicates (distinct)", all(i in dups for i in ids) and len(set(ids)) == len(ids)),
            (f"{label}: newest order kept", keep is None or str(keep) not in ids)]


def _order_checks(pl, ref, rtype, items):
    base = [("resourceType", pl is not None and pl.get("resourceType") == rtype),
            ("status active", pl is not None and str(pl.get("status", "")).lower() == "active"),
            ("intent order", pl is not None and str(pl.get("intent", "")).lower() == "order"),
            ("subject", pl is not None and pl.get("subject") == ref)]
    return base + [(n, pl is not None and R_ok(f, pl)) for n, f in items]


def R_ok(f, pl):
    try:
        return bool(f(pl))
    except Exception:
        return False


def _first(posts, pred):
    return next((pl for _, pl in posts if R_ok(pred, pl)), None)


def score(case_data, results, fhir_api_base, full: bool):
    t = case_data["id"].split("_")[0]
    exp = getattr(R, f"ref_{t}")(case_data, fhir_api_base)
    mrn, ref = case_data["eval_MRN"], {"reference": f"Patient/{case_data['eval_MRN']}"}
    now = R.extract_now_from_context(case_data)
    posts = R.extract_posts(results)
    if t in R.LAB:                                         # lab-threshold types (task7 - task12)
        spec = R.LAB[t]
        g = {"query": [(f"GET Observation {spec['code']}", _fetched(results, "Observation", mrn, [spec["code"]]))]}
        if exp["kind"] == "silent":
            g["action"] = [("post nothing", not R.check_has_post(results))]
        else:
            a = [(f"exactly {len(spec['orders'])} POST(s)", len(posts) == len(spec["orders"]))]
            for o in spec["orders"]:
                best = max(posts, key=lambda p: sum(ok for _, ok in R.lab_order_checks(p[1], ref, o, now)), default=None)
                a += R.lab_order_checks(best[1] if best else None, ref, o, now)
            g["action"] = a
        g["action"].append(("called FINISH", results.result is not None))
        fracs = [sum(ok for _, ok in c) / len(c) for c in g.values() if c]
        frac = sum(fracs) / len(fracs)
        return {"score": 1.0 if full else round(0.5 * frac, 4), "full": bool(full), "frac": round(frac, 4),
                "groups": {k: [[n, bool(ok)] for n, ok in c] for k, c in g.items()}}
    g = {"query": [("GET MedicationRequest", _fetched(results, "MedicationRequest", mrn))]}
    if t in ("task5", "task6"):
        g["query"].append(("GET Condition", _fetched(results, "Condition", mrn)))
    if t == "task2" and exp.get("kind") != "dedupe" and exp.get("reason") != "one active order":
        g["query"].append(("GET Observation MG", _fetched(results, "Observation", mrn, ["MG"])))
    if exp["kind"] == "silent":
        g["action"] = [("post nothing", not R.check_has_post(results))]
    elif exp["kind"] == "dedupe":
        if t == "task3":
            a = []
            n = 0
            for drug, key, label in ((R.SENNA, "senna", "sennosides"), ("polyethylene glycol", "peg", "PEG 3350")):
                if exp[f"{key}_dups"]:
                    a += _dedupe_checks(posts, ref, drug, exp[f"{key}_dups"], exp[f"{key}_keep"], label)
                    n += len(exp[f"{key}_dups"])
            a.append(("nothing else posted", len(posts) == n))
        else:
            a = _dedupe_checks(posts, ref, exp["drug"], exp["dups"], exp["keep"], exp["drug"])
            a.append(("nothing else posted", len(posts) == len(exp["dups"])))
        a.append(("every POST for this patient", all(pl.get("subject") == ref for _, pl in posts)))
        g["action"] = a
    else:   # order
        if t == "task2":
            pl = _first(posts, lambda p: R.mentions(p, "magnesium sulfate"))
            items = [("magnesium sulfate", lambda p: R.mentions(p, "magnesium sulfate")),
                     ("dose 2 g", lambda p: R.dose_is(p, 2, ("g", "gram"))), ("route IV", R.is_iv),
                     ("authored now", lambda p: R.authored_now(p, now))]
            g["action"] = [("exactly 1 POST", len(posts) == 1)] + _order_checks(pl, ref, "MedicationRequest", items)
        elif t == "task3":
            pl = _first(posts, lambda p: R.mentions(p, "polyethylene glycol"))
            items = [("polyethylene glycol 3350", lambda p: R.mentions(p, "polyethylene glycol")),
                     ("dose 17 g", lambda p: R.dose_is(p, 17, ("g", "gram"))), ("route PO", R.is_oral),
                     ("daily", lambda p: R.every_hours(p, 24)), ("authored now", lambda p: R.authored_now(p, now))]
            g["action"] = [("exactly 1 POST", len(posts) == 1)] + _order_checks(pl, ref, "MedicationRequest", items)
        elif t == "task4":
            pl = _first(posts, lambda p: R.mentions(p, "heparin"))
            items = [("heparin", lambda p: R.mentions(p, "heparin")),
                     ("dose 5,000 units", lambda p: R.dose_is(p, 5000, ("unit",)) or R.mentions(p, "5,000") or R.mentions(p, "5000")),
                     ("route SC", R.is_subcutaneous), ("every 8 hours", lambda p: R.every_hours(p, 8)),
                     ("authored now", lambda p: R.authored_now(p, now))]
            g["action"] = [("exactly 1 POST", len(posts) == 1)] + _order_checks(pl, ref, "MedicationRequest", items)
        elif t == "task5":
            pl = _first(posts, lambda p: R.mentions(p, "rosuvastatin"))
            items = [("rosuvastatin", lambda p: R.mentions(p, "rosuvastatin")), ("dose 40 mg", lambda p: R.dose_is(p, 40, ("mg",))),
                     ("route PO", R.is_oral), ("daily", lambda p: R.every_hours(p, 24)),
                     ("authored now", lambda p: R.authored_now(p, now))]
            g["action"] = [("exactly 1 POST", len(posts) == 1)] + _order_checks(pl, ref, "MedicationRequest", items)
        else:   # task6
            med = _first(posts, lambda p: R.mentions(p, "metoprolol"))
            ecg = _first(posts, lambda p: p.get("resourceType") == "ServiceRequest")
            med_items = [("metoprolol succinate", lambda p: R.mentions(p, "metoprolol succinate")),
                         ("dose 25 mg", lambda p: R.dose_is(p, 25, ("mg",))), ("route PO", R.is_oral),
                         ("daily", lambda p: R.every_hours(p, 24)), ("authored now", lambda p: R.authored_now(p, now))]
            ecg_items = [("ECG SNOMED 445118002", lambda p: R.mentions(p, "445118002")),
                         ("ECG 30 minutes after the dose",
                          lambda p: (now + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M") in str(p.get("occurrenceDateTime", "")))]
            g["action"] = ([("exactly 2 POSTs", len(posts) == 2)]
                           + [(f"med: {n}", ok) for n, ok in _order_checks(med, ref, "MedicationRequest", med_items)]
                           + [(f"ECG: {n}", ok) for n, ok in _order_checks(ecg, ref, "ServiceRequest", ecg_items)])
    g["action"].append(("called FINISH", results.result is not None))
    fracs = [sum(ok for _, ok in c) / len(c) for c in g.values() if c]
    frac = sum(fracs) / len(fracs) if fracs else 0.0
    return {"score": 1.0 if full else round(0.5 * frac, 4), "full": bool(full), "frac": round(frac, 4),
            "groups": {k: [[n, bool(ok)] for n, ok in c] for k, c in g.items()}}
