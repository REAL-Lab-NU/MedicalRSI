"""Process score for MedAgentBench v2, as for v1: query (the data the reference decision reads), action (the decision,
every condition on the POSTs, and "called FINISH") and answer (the returned values of task3). All checks passing implies
that the reference grader passes."""
import ast
import json
from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

from .refsol import CUTOFF, before_cutoff, extract_now_from_context
from .utils import send_get_request


def _role(m):
    return m.role if hasattr(m, "role") else m.get("role")


def _content(m):
    return (m.content if hasattr(m, "content") else m.get("content")) or ""


def _agent(results):
    return [_content(m) for m in results.history if _role(m) == "agent"]


def _has_post(results):
    """refsol.check_has_post: any agent message containing 'POST'."""
    return any("POST" in c for c in _agent(results))


def _posts(results):
    """refsol.extract_posts: accepted POSTs as (url, payload)."""
    h, out = results.history, []
    for i, m in enumerate(h):
        if _role(m) == "agent" and "POST" in _content(m):
            if i + 1 < len(h) and "POST request accepted" in _content(h[i + 1]):
                try:
                    r = _content(m)
                    out.append((r.split("\n")[0][4:].strip(), json.loads("\n".join(r.split("\n")[1:]))))
                except Exception:
                    pass
    return out


def _fetched(results, resource, mrn, codes=None):
    """A GET of `resource` for this patient; if the agent filtered by code, the filter includes a reference code."""
    want = {c.lower() for c in (codes or [])}
    for c in _agent(results):
        c = c.strip()
        if not c.startswith("GET"):
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


def _ok(fn):
    try:
        return bool(fn())
    except Exception:
        return False


def _get(base, query):
    return json.loads(send_get_request(f"{base}{query}&_count=5000&_format=json")["data"]).get("entry", [])


def _performed(res):
    return res.get("performedDateTime") or res.get("performedPeriod", {}).get("end") or res.get("performedPeriod", {}).get("start")


def _latest(entries):
    last = None
    for e in entries:
        dt = _performed(e["resource"])
        if dt:
            ts = datetime.fromisoformat(dt)
            if last is None or ts > last:
                last = ts
    return last


def _silent(posts):
    return [("post nothing", not posts)]


def _order_form(pl, ref, types=("ServiceRequest", "MedicationRequest")):
    return [("resourceType", pl is not None and pl.get("resourceType") in types),
            ("status active", pl is not None and _ok(lambda: pl.get("status", "").lower() == "active")),
            ("intent order", pl is not None and _ok(lambda: pl.get("intent", "").lower() == "order")),
            ("subject", pl is not None and pl.get("subject") == ref)]


def _best(posts, checks):
    """The POST that passes most of `checks` (functions of the payload); None if there is no POST."""
    best, n = None, -1
    for _, pl in posts:
        k = sum(_ok(lambda f=f: f(pl)) for f in checks)
        if k > n:
            best, n = pl, k
    return best


def t1(c, r, base):
    mrn, ref = c["eval_MRN"], {"reference": f"Patient/{c['eval_MRN']}"}
    entries = [e for e in _get(base, f"Procedure?patient={mrn}&code=IMGCT0491,IMGIL0001") if before_cutoff(e["resource"])]
    last = _latest(entries)
    required = last is None or last < extract_now_from_context(c) - timedelta(days=365)   # as in refsol
    g = {"query": [("GET Procedure (prior CT)", _fetched(r, "Procedure", mrn, ["IMGCT0491", "IMGIL0001"]))]}
    if not required:
        g["action"] = [("post nothing (recent CT)", not _has_post(r))]
        return g
    posts = _posts(r)

    def indication(pl):
        txt = [x.get("text", "") for x in pl.get("reasonCode", [])] + [x.get("display", "") for x in pl.get("reasonReference", [])]
        notes = pl.get("note", [])
        txt += [n.get("text", "") for n in ([notes] if isinstance(notes, dict) else notes)]
        return any("renal mass follow-up" in t.lower() for t in txt)
    pl = posts[0][1] if posts else None
    g["action"] = [("exactly 1 POST", len(posts) == 1),
                   ("POST to ServiceRequest", bool(posts) and "ServiceRequest" in posts[0][0]),
                   ("resourceType", pl is not None and pl.get("resourceType") == "ServiceRequest"),
                   ("status active", pl is not None and _ok(lambda: pl["status"].lower() == "active")),
                   ("intent order", pl is not None and _ok(lambda: pl["intent"].lower() == "order")),
                   ("subject", pl is not None and pl.get("subject") == ref),
                   ("CPT 74177", pl is not None and _ok(lambda: pl["code"]["coding"][0]["code"] == "74177")),
                   ("indication renal mass follow-up", pl is not None and _ok(lambda: indication(pl)))]
    return g


def t2(c, r, base):
    mrn, ref = c["eval_MRN"], {"reference": f"Patient/{c['eval_MRN']}"}
    keys = ["heparin", "enoxaparin", "dalteparin", "fondaparinux", "apixaban", "rivaroxaban", "dabigatran", "warfarin"]
    txt = lambda e: e["resource"]["medicationCodeableConcept"].get("text", "").lower()
    entries = [e for e in _get(base, f"MedicationRequest?patient={mrn}&status=active") if before_cutoff(e["resource"])]
    acs = [e for e in entries if any(k in txt(e) for k in keys)]
    heps = [e for e in acs if "heparin" in txt(e) and ("5,000" in txt(e) or "5000" in txt(e))]
    need_create, need_stops = len(heps) == 0, max(len(acs) - 1, 0)
    posts = _posts(r)
    g = {"query": [("GET MedicationRequest", _fetched(r, "MedicationRequest", mrn))]}
    if not need_create and need_stops == 0:
        g["action"] = _silent(posts)
        return g
    creates = [p for p in posts if p[1].get("resourceType") == "MedicationRequest" and p[1].get("status", "").lower() == "active"]
    stops = [p for p in posts if p[1].get("resourceType") == "MedicationRequest"
             and p[1].get("status", "").lower() in {"stopped", "cancelled", "completed", "entered-in-error"}]
    a = []
    if need_create:
        a.append(("new active heparin 5000 order", any("heparin" in str(pl).lower() and "5000" in str(pl).lower() for _, pl in creates)))
    if need_stops > 0:
        a.append(("extra anticoagulant discontinued", bool(stops)))
    a.append(("every POST for this patient", all(pl.get("subject") == ref for _, pl in posts)))
    g["action"] = a
    return g


def t3(c, r, base):
    mrn = c["eval_MRN"]
    now = extract_now_from_context(c)
    v6, v12 = [], []
    for e in _get(base, f"Observation?patient={mrn}&code=HEARTRATE"):
        res = e.get("resource", {})
        if not before_cutoff(res):
            continue
        try:
            ts = datetime.fromisoformat(res["effectiveDateTime"])
            bpm = res["valueQuantity"]["value"]
        except Exception:
            continue
        if now - timedelta(hours=12) <= ts <= now:
            v12.append(bpm)
            if now - timedelta(hours=6) <= ts:
                v6.append(bpm)
    gold = [sum(v) / len(v) if v else None for v in (v6, v12)]
    raw = r.result
    parsed = None
    if isinstance(raw, (list, tuple)):
        parsed = raw
    else:
        for parse in (ast.literal_eval, json.loads):
            try:
                parsed = parse(raw)
                break
            except Exception:
                continue
    two = isinstance(parsed, (list, tuple)) and len(parsed) == 2
    norm = lambda v: None if v is None or v == -1 or (isinstance(v, str) and v.strip() == "-1") else v

    def close(a, b):
        a = norm(a)
        if a is None and b is None:
            return True
        if (a is None) ^ (b is None):
            return False
        try:
            return abs(float(a) - float(b)) <= 0.1
        except Exception:
            return False
    return {"query": [("GET Observation HEARTRATE", _fetched(r, "Observation", mrn, ["HEARTRATE"]))],
            "action": [("no POST (read-only)", not _has_post(r))],
            "answer": [("two values returned", two),
                       (f"6h mean (±0.1) (expected {gold[0]})", two and close(parsed[0], gold[0])),
                       (f"12h mean (±0.1) (expected {gold[1]})", two and close(parsed[1], gold[1]))]}


def t4(c, r, base):
    mrn, ref = c["eval_MRN"], {"reference": f"Patient/{c['eval_MRN']}"}
    now = extract_now_from_context(c)
    inserts = [e for e in _get(base, f"Procedure?patient={mrn}&code=NUR1373") if before_cutoff(e["resource"])]
    last = _latest(inserts)
    removals = [e for e in _get(base, f"ServiceRequest?patient={mrn}&code=NUR1373&status=active") if before_cutoff(e["resource"])]
    required = last is not None and now - last > timedelta(hours=48) and not removals
    posts = _posts(r)
    g = {"query": [("GET Procedure (catheter)", _fetched(r, "Procedure", mrn, ["NUR1373"]))]}
    if not required:
        g["action"] = _silent(posts)
        return g
    checks = [lambda p: p.get("resourceType") == "ServiceRequest", lambda p: p.get("status", "").lower() == "active",
              lambda p: p.get("intent", "").lower() == "order", lambda p: p.get("subject") == ref,
              lambda p: "nur1373" in str(p.get("code", "")).lower()]
    pl = _best(posts, checks)
    g["action"] = [("at least 1 POST", bool(posts))] + [
        (n, pl is not None and _ok(lambda f=f: f(pl))) for n, f in
        zip(("resourceType ServiceRequest", "status active", "intent order", "subject", "code NUR1373"), checks)]
    return g


def t5(c, r, base):
    mrn, ref = c["eval_MRN"], {"reference": f"Patient/{c['eval_MRN']}"}
    now = extract_now_from_context(c)
    dx = [e for e in _get(base, f"Condition?patient={mrn}&code=C64.2") if before_cutoff(e["resource"])]
    g = {"query": [("GET Condition (C64.2)", _fetched(r, "Condition", mrn, ["C64.2"]))]}
    if not dx:
        g["action"] = [("post nothing (no diagnosis)", not _has_post(r))]
        return g
    g["query"].append(("GET Procedure (prior CT)", _fetched(r, "Procedure", mrn, ["IMGCT0491", "IMGIL0001", "74177"])))
    cts = [e for e in _get(base, f"Procedure?patient={mrn}&code=IMGCT0491,IMGIL0001,74177") if before_cutoff(e["resource"])]
    last = _latest(cts)
    posts = _posts(r)
    if not (last is None or now - last > timedelta(days=90)):
        g["action"] = _silent(posts)
        return g
    valid = [pl for _, pl in posts if pl.get("resourceType") in {"ServiceRequest", "MedicationRequest"}
             and pl.get("status", "").lower() == "active" and pl.get("intent", "").lower() == "order" and pl.get("subject") == ref]
    g["action"] = [("a well-formed active order for this patient", bool(valid)),
                   ("CT 74177 ordered", any("74177" in str(pl.get("code", {})) for pl in valid)),
                   ("IR CON417 ordered", any("CON417" in str(pl.get("code", {})) for pl in valid))]
    return g


def t6(c, r, base):
    mrn, ref = c["eval_MRN"], {"reference": f"Patient/{c['eval_MRN']}"}

    def obs(code):
        out = []
        for e in _get(base, f"Observation?patient={mrn}&code={code}"):
            if before_cutoff(e["resource"]):
                out.append((e["resource"]["valueQuantity"]["value"], datetime.fromisoformat(e["resource"]["effectiveDateTime"])))
        return out
    tsh, ft4 = obs("TSH"), obs("FT4")
    last_tsh = max(tsh, key=lambda x: x[1])[0] if tsh else None
    last_ft4 = max(ft4, key=lambda x: x[1])[0] if ft4 else None
    high = sorted(d for v, d in tsh if v > 10)
    branch_a = any((high[j] - high[i]).days >= 30 for i in range(len(high)) for j in range(i + 1, len(high)))
    branch_b = last_tsh is not None and 4.5 <= last_tsh <= 10 and last_ft4 is not None and 0.8 <= last_ft4 <= 1.8
    posts = _posts(r)
    meds = [p for p in posts if p[1].get("resourceType") == "MedicationRequest"]
    labs = [p for p in posts if p[1].get("resourceType") == "ServiceRequest"]
    def ok_med(p):    # refsol's ok_med; like refsol, a payload that makes it raise fails the whole check
        txt = p[1]["medicationCodeableConcept"].get("text", "").lower()
        return (p[1].get("status", "").lower() == "active" and p[1].get("intent", "").lower() == "order"
                and p[1].get("subject") == ref and "levothyroxine" in txt and "25" in txt)

    def ok_lab(p):
        txt = str(p[1].get("code", {})).lower() + p[1].get("code", {}).get("text", "").lower()
        return (p[1].get("status", "").lower() == "active" and p[1].get("intent", "").lower() == "order"
                and p[1].get("subject") == ref and ("tsh" in txt or "ft4" in txt))

    def strict(fn, fail):
        try:
            return fn()
        except Exception:
            return fail
    any_med = strict(lambda: any(ok_med(p) for p in meds), False)
    n_lab = strict(lambda: sum(ok_lab(p) for p in labs), -1)
    g = {"query": [("GET Observation TSH", _fetched(r, "Observation", mrn, ["TSH"])),
                   ("GET Observation FT4", _fetched(r, "Observation", mrn, ["FT4"]))]}
    if branch_a:
        g["action"] = [("levothyroxine 25 order", any_med),
                       ("1st repeat TSH/FT4 order", n_lab >= 1), ("2nd repeat TSH/FT4 order", n_lab >= 2)]
    elif branch_b:
        g["action"] = [("no medication order", not meds),
                       ("1st repeat TSH/FT4 order", n_lab >= 1), ("2nd repeat TSH/FT4 order", n_lab >= 2)]
    else:
        g["action"] = _silent(posts)
    return g


def t7(c, r, base):
    mrn, ref = c["eval_MRN"], {"reference": f"Patient/{c['eval_MRN']}"}
    words = ["ondansetron", "prochlorperazine", "haloperidol", "quetiapine", "olanzapine", "risperidone", "ziprasidone",
             "clozapine", "chlorpromazine"]
    qt = [e for e in _get(base, f"Observation?patient={mrn}&code=QTCINTERVAL") if before_cutoff(e["resource"])]
    g = {"query": [("GET Observation QTc", _fetched(r, "Observation", mrn, ["QTCINTERVAL"]))]}
    if not qt:
        g["action"] = [("post nothing (no QTc)", not _has_post(r))]
        return g
    latest = max(qt, key=lambda e: e["resource"]["meta"].get("lastUpdated", ""))["resource"]
    posts = _posts(r)
    if not latest["valueQuantity"]["value"] > 500:
        g["action"] = _silent(posts)
        return g
    ecg = any(pl.get("resourceType") == "ServiceRequest" and pl.get("status", "").lower() == "active"
              and ("445118002" in str(pl).lower() or "ecg" in str(pl).lower()) for _, pl in posts)
    stop = any(pl.get("resourceType") == "MedicationRequest"
               and pl.get("status", "").lower() in {"stopped", "cancelled", "completed", "entered-in-error"}
               and any(w in str(pl).lower() for w in words) for _, pl in posts)
    g["action"] = [("every POST a Service/MedicationRequest", all(pl.get("resourceType") in {"ServiceRequest", "MedicationRequest"} for _, pl in posts)),
                   ("every POST for this patient", all(pl.get("subject") == ref for _, pl in posts)),
                   ("12-lead ECG ordered", ecg), ("QT-prolonging drug discontinued", stop)]
    return g


def _one_order(posts, ref, mention, words):
    pl = posts[0][1] if posts else None
    return ([("exactly 1 POST", len(posts) == 1)] + _order_form(pl, ref)
            + [(mention, pl is not None and any(w in str(pl).lower() for w in words))])


def t8(c, r, base):
    mrn, ref = c["eval_MRN"], {"reference": f"Patient/{c['eval_MRN']}"}
    txt = lambda e: e["resource"]["medicationCodeableConcept"].get("text", "").lower()
    base_meds = [e for e in _get(base, f"MedicationRequest?patient={mrn}&status=active")
                 if e["resource"]["meta"].get("lastUpdated") is None or datetime.fromisoformat(e["resource"]["meta"]["lastUpdated"]) <= CUTOFF]
    opioids = [e for e in base_meds if any(w in txt(e) for w in ["hydromorphone", "oxycodone", "fentanyl", "hydrocodone", "morphine"])]
    nalox = [e for e in base_meds if "naloxone" in txt(e) or "narcan" in txt(e)]
    posts = _posts(r)
    g = {"query": [("GET MedicationRequest", _fetched(r, "MedicationRequest", mrn))]}
    g["action"] = _one_order(posts, ref, "mentions naloxone/narcan", ["naloxone", "narcan"]) if (opioids and not nalox) else _silent(posts)
    return g


def t9(c, r, base):
    mrn, ref = c["eval_MRN"], {"reference": f"Patient/{c['eval_MRN']}"}
    now = extract_now_from_context(c)
    shots = [e for e in _get(base, f"Procedure?patient={mrn}&code=90686") if before_cutoff(e["resource"])]
    last = _latest(shots)
    posts = _posts(r)
    g = {"query": [("GET Procedure (flu shots)", _fetched(r, "Procedure", mrn, ["90686"]))]}
    if not (last is None or now - last > timedelta(days=365)):
        g["action"] = _silent(posts)
        return g
    pl = posts[0][1] if posts else None

    def flu(pl):
        ok = False
        if "code" in pl and "coding" in pl["code"]:
            ok = pl["code"]["coding"][0].get("code") == "90686"
        return ok or "flu" in str(pl).lower() or "influenza" in str(pl).lower()
    g["action"] = [("exactly 1 POST", len(posts) == 1)] + _order_form(pl, ref) + [
        ("recognisably a flu shot", pl is not None and _ok(lambda: flu(pl)))]
    return g


def t10(c, r, base):
    mrn, ref = c["eval_MRN"], {"reference": f"Patient/{c['eval_MRN']}"}
    now = extract_now_from_context(c)
    procs = [e for e in _get(base, f"Procedure?patient={mrn}&code=COVIDVACCINE") if before_cutoff(e["resource"])]
    meds = [e for e in _get(base, f"MedicationRequest?patient={mrn}&status=completed")
            if before_cutoff(e["resource"]) and "covid-19 vac" in e["resource"]["medicationCodeableConcept"].get("text", "").lower()]
    dates = []
    for e in procs + meds:
        dt = _performed(e["resource"]) or e["resource"].get("authoredOn")
        if dt:
            dates.append(datetime.fromisoformat(dt))
    last = max(dates) if dates else None
    needs = last is None or (now.date() - last.date()).days > 365
    posts = _posts(r)
    g = {"query": [("GET prior COVID vaccinations", _fetched(r, "Procedure", mrn, ["COVIDVACCINE"])
                    or _fetched(r, "MedicationRequest", mrn))]}
    if not needs:
        g["action"] = _silent(posts)
        return g
    pl = posts[0][1] if posts else None
    g["action"] = [("exactly 1 POST", len(posts) == 1)] + _order_form(pl, ref) + [
        ("mentions COVID", pl is not None and "covid" in (str(pl.get("code", {})) + str(pl.get("medicationCodeableConcept", {}))).lower())]
    return g


CHECKS = {f"task{i}": f for i, f in enumerate((t1, t2, t3, t4, t5, t6, t7, t8, t9, t10), 1)}


def score(case_data, results, fhir_api_base, full: bool):
    """{'score', 'full', 'frac', 'groups'}; full is the original grader's verdict on this episode."""
    groups = CHECKS[case_data["id"].split("_")[0]](case_data, results, fhir_api_base)
    groups.setdefault("action", []).append(("called FINISH", results.result is not None))
    fracs = [sum(ok for _, ok in g) / len(g) for g in groups.values() if g]
    frac = sum(fracs) / len(fracs) if fracs else 0.0
    return {"score": 1.0 if full else round(0.5 * frac, 4), "full": bool(full), "frac": round(frac, 4),
            "groups": {k: [[n, bool(ok)] for n, ok in g] for k, g in groups.items()}}
