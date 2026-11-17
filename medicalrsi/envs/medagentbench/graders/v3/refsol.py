"""Reference graders of MedAgentBench v3. The first six task types are built from the unused physician
draft templates of MedAgentBench V2 (github.com/ericoericochen/medagentbenchv2, data/medagentbench/new_data_v1.json:
task14, task23, task29, task30, task12, task13), graded in the style of MedAgentBench V2's new_refsol.py: the baseline
chart is the FHIR store as of the cut-off (meta.lastUpdated <= 2025-01-01), "now" comes from the task context, and only
the clinically essential parts of each POST are checked (resource type, status, intent, patient, drug / code, dose,
route, frequency, timing), accepting the usual FHIR spellings of route and frequency.

  task1  polyethylene glycol 3350 duplicates   >1 active order -> discontinue all but the most recently authored one
  task2  magnesium sulfate IV                  >1 active -> discontinue all but the newest; none active and the last
                                               serum Mg within 24 h < 1.8 mg/dL -> order magnesium sulfate 2 g IV
  task3  laxatives (sennosides, PEG 3350)      per agent >1 active -> keep only the newest; neither active ->
                                               order PEG 3350 17 g PO daily
  task4  VTE prophylaxis                       no active anticoagulant -> heparin 5,000 units SC q8h now
  task5  statin for hyperlipidemia (E78.*)     diagnosis and no active statin -> rosuvastatin 40 mg PO daily
  task6  beta-blocker for AF (I48.91)          AF and no active metoprolol / carvedilol -> metoprolol succinate 25 mg
                                               PO daily + 12-lead ECG (SNOMED 445118002) 30 minutes later

A discontinuation is a POSTed MedicationRequest for the patient, mentioning the drug, with a non-active status
(stopped / cancelled / completed / entered-in-error); when it carries an id, the id must be one of the orders to stop.
Every ref_taskN(case_data, fhir_api_base) returns the expected behaviour; taskN grades an episode against it and
refsol_partial.py turns the same expectations into partial credit.
"""
import json
import re
from datetime import datetime, timedelta

from .utils import *

CUTOFF = datetime.fromisoformat("2025-01-01T00:00:00+00:00")
DEFAULT_NOW = datetime.fromisoformat("2023-11-13T10:15:00+00:00")
NON_ACTIVE = {"stopped", "cancelled", "completed", "entered-in-error"}
ANTICOAG = ["heparin", "enoxaparin", "dalteparin", "fondaparinux", "apixaban", "rivaroxaban", "dabigatran", "warfarin"]
STATINS = ["atorvastatin", "rosuvastatin", "simvastatin", "pravastatin", "lovastatin", "pitavastatin", "fluvastatin"]
BETA_BLOCKERS = ["metoprolol", "carvedilol"]
PEG, SENNA, MGSO4 = "polyethylene glycol 3350", "sennosides", "magnesium sulfate"


# ---------------------------------------------------------------- shared helpers (as in MedAgentBench V2)

def extract_now_from_context(case_data):
    m = re.search(r"It'?s\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?[+-]\d{2}:\d{2})\s*now", case_data.get("context", "") or "",
                  re.IGNORECASE)
    return datetime.fromisoformat(m.group(1)) if m else DEFAULT_NOW


def before_cutoff(res):
    ts = (res.get("meta") or {}).get("lastUpdated")
    return ts is None or datetime.fromisoformat(ts) <= CUTOFF


def extract_posts(results):
    posts = []
    h = results.history
    for idx, i in enumerate(h):
        if (i.role == "agent") and ("POST" in i.content) and idx + 1 < len(h) and "POST request accepted" in h[idx + 1].content:
            try:
                r = i.content
                posts.append((r.split("\n")[0][4:].strip(), json.loads("\n".join(r.split("\n")[1:]))))
            except Exception:
                pass
    return posts


def check_has_post(results):
    return any((i.role == "agent") and ("POST" in i.content) for i in results.history)


def _bundle(fhir_api_base, query):
    return json.loads(send_get_request(f"{fhir_api_base}{query}&_count=5000&_format=json")["data"]).get("entry", [])


def _active_meds(case_data, fhir_api_base):
    return [e["resource"] for e in _bundle(fhir_api_base, f"MedicationRequest?patient={case_data['eval_MRN']}&status=active")
            if before_cutoff(e["resource"])]


def _med_text(r):
    return (r.get("medicationCodeableConcept") or {}).get("text", "").lower()


def _conditions(case_data, fhir_api_base):
    out = []
    for e in _bundle(fhir_api_base, f"Condition?patient={case_data['eval_MRN']}"):
        if before_cutoff(e["resource"]):
            out += [c.get("code", "") for c in (e["resource"].get("code") or {}).get("coding", [])]
    return out


def _newest(orders):
    return max(orders, key=lambda r: (r.get("authoredOn") or "", str(r.get("id"))))


def _dups(orders):
    """Ids of the orders to discontinue: all but the most recently authored one (none when <= 1 order)."""
    if len(orders) <= 1:
        return []
    keep = _newest(orders)["id"]
    return sorted(str(r["id"]) for r in orders if r["id"] != keep)


# ---------------------------------------------------------------- payload checks (essential, spelling-tolerant)

def _route(pl):
    try:
        r = pl["dosageInstruction"][0]["route"]
    except Exception:
        return ""
    if isinstance(r, str):
        return r.lower()
    if isinstance(r, dict):
        c = (r.get("coding") or [{}])[0]
        return " ".join(str(x) for x in (r.get("text"), c.get("display"), c.get("code")) if x).lower()
    return ""


def is_oral(pl):
    s = _route(pl)
    return "oral" in s or "mouth" in s or re.search(r"\bpo\b", s) is not None or s.strip() == "po"


def is_iv(pl):
    s = _route(pl)
    return "intravenous" in s or re.search(r"\biv\b", s) is not None


def is_subcutaneous(pl):
    s = _route(pl)
    return "subcutaneous" in s or re.search(r"\b(sc|subq|sq)\b", s) is not None


def _timing(pl):
    try:
        return pl["dosageInstruction"][0]["timing"]
    except Exception:
        return {}


def every_hours(pl, hours):
    """Frequency check: timing code text (q8h / every 8 hours / daily ...) or timing.repeat."""
    t = _timing(pl)
    txt = json.dumps(t).lower() + " " + str((pl.get("dosageInstruction") or [{}])[0].get("text", "")).lower()
    if hours == 24 and re.search(r"daily|once a day|\bqd\b|q24h|every 24 hours|once daily", txt):
        return True
    if hours != 24 and re.search(rf"q{hours}h|q {hours} ?h|every {hours} hours", txt):
        return True
    rep = t.get("repeat") or {}
    try:
        f, p, u = float(rep.get("frequency", 1)), float(rep.get("period")), str(rep.get("periodUnit", "")).lower()
        per_hour = {"h": 1, "d": 24, "min": 1 / 60}.get(u)
        return per_hour is not None and abs(p * per_hour / f - hours) < 1e-6
    except Exception:
        return False


def mentions(pl, *words):
    s = json.dumps(pl).lower()
    return all(w.lower() in s for w in words)


def dose_is(pl, value, units):
    """dose via doseAndRate[0].doseQuantity, or the strength written in the medication text ('40 MG', '5,000 UNIT')."""
    try:
        q = pl["dosageInstruction"][0]["doseAndRate"][0]["doseQuantity"]
        if abs(float(q["value"]) - value) < 1e-6 and str(q.get("unit", "")).lower().rstrip("s") in units:
            return True
    except Exception:
        pass
    txt = (pl.get("medicationCodeableConcept") or {}).get("text", "").lower()
    forms = {f"{value:g}"} | ({f"{int(value):,}"} if value >= 1000 and float(value).is_integer() else set())
    return any(re.search(rf"(?<![\d.,]){re.escape(f)}\s*{u}", txt) for f in forms for u in units)


def authored_now(pl, now):
    return now.strftime("%Y-%m-%dT%H:%M") in str(pl.get("authoredOn", ""))


def order_ok(pl, ref, rtype="MedicationRequest"):
    return (pl.get("resourceType") == rtype and str(pl.get("status", "")).lower() == "active"
            and str(pl.get("intent", "")).lower() == "order" and pl.get("subject") == ref)


def is_discontinuation(pl, ref, drug):
    return (pl.get("resourceType") == "MedicationRequest" and pl.get("subject") == ref
            and str(pl.get("status", "")).lower() in NON_ACTIVE and drug in json.dumps(pl).lower())


def _stops_ok(posts, ref, drug, dup_ids):
    """The discontinuations of `drug` stop exactly the duplicates: one per duplicate; ids, when given, are duplicates and
    distinct."""
    stops = [pl for _, pl in posts if is_discontinuation(pl, ref, drug)]
    ids = [str(pl["id"]) for pl in stops if pl.get("id") is not None]
    return (len(stops) == len(dup_ids) and all(i in dup_ids for i in ids) and len(set(ids)) == len(ids))


# ---------------------------------------------------------------- expectations per task

def ref_task1(c, base):
    peg = [r for r in _active_meds(c, base) if PEG in _med_text(r)]
    return {"kind": "dedupe" if len(peg) > 1 else "silent", "drug": "polyethylene glycol", "dups": _dups(peg),
            "keep": _newest(peg)["id"] if peg else None}


def _last_mg_24h(c, base, now):
    last_t, last_v = None, None
    for e in _bundle(base, f"Observation?patient={c['eval_MRN']}&code=MG"):
        r = e["resource"]
        if not before_cutoff(r):
            continue
        t = datetime.fromisoformat(r["effectiveDateTime"])
        if now - timedelta(hours=24) <= t <= now and (last_t is None or t > last_t):
            last_t, last_v = t, r["valueQuantity"]["value"]
    return last_v


def ref_task2(c, base):
    now = extract_now_from_context(c)
    mg = [r for r in _active_meds(c, base) if MGSO4 in _med_text(r)]
    if len(mg) > 1:
        return {"kind": "dedupe", "drug": MGSO4, "dups": _dups(mg), "keep": _newest(mg)["id"]}
    if len(mg) == 1:
        return {"kind": "silent", "reason": "one active order"}
    v = _last_mg_24h(c, base, now)
    if v is not None and v < 1.8:
        return {"kind": "order", "last_mg": v}
    return {"kind": "silent", "reason": "no order, Mg not low or not measured"}


def ref_task3(c, base):
    meds = _active_meds(c, base)
    senna = [r for r in meds if SENNA in _med_text(r)]
    peg = [r for r in meds if PEG in _med_text(r)]
    out = {"senna_dups": _dups(senna), "peg_dups": _dups(peg), "order_peg": not senna and not peg,
           "senna_keep": _newest(senna)["id"] if senna else None, "peg_keep": _newest(peg)["id"] if peg else None}
    out["kind"] = "order" if out["order_peg"] else "dedupe" if (out["senna_dups"] or out["peg_dups"]) else "silent"
    return out


def ref_task4(c, base):
    acs = [r for r in _active_meds(c, base) if any(w in _med_text(r) for w in ANTICOAG)]
    return {"kind": "silent" if acs else "order"}


def ref_task5(c, base):
    hl = any(code.startswith("E78") for code in _conditions(c, base))
    statin = any(any(w in _med_text(r) for w in STATINS) for r in _active_meds(c, base))
    return {"kind": "order" if hl and not statin else "silent", "hyperlipidemia": hl, "statin": statin}


def ref_task6(c, base):
    af = "I48.91" in _conditions(c, base)
    bb = any(any(w in _med_text(r) for w in BETA_BLOCKERS) for r in _active_meds(c, base))
    return {"kind": "order" if af and not bb else "silent", "af": af, "beta_blocker": bb}


# ---------------------------------------------------------------- graders

def _ref(c):
    return {"reference": f"Patient/{c['eval_MRN']}"}


def task1(case_data, results, fhir_api_base):
    exp, posts = ref_task1(case_data, fhir_api_base), extract_posts(results)
    if exp["kind"] == "silent":
        return not check_has_post(results)
    return bool(posts) and len(posts) == len(exp["dups"]) and _stops_ok(posts, _ref(case_data), exp["drug"], exp["dups"])


def magnesium_order_ok(pl, ref, now):
    return (order_ok(pl, ref) and mentions(pl, "magnesium sulfate") and dose_is(pl, 2, ("g", "gram"))
            and is_iv(pl) and authored_now(pl, now))


def task2(case_data, results, fhir_api_base):
    exp, posts, ref = ref_task2(case_data, fhir_api_base), extract_posts(results), _ref(case_data)
    if exp["kind"] == "silent":
        return not check_has_post(results)
    if exp["kind"] == "dedupe":
        return bool(posts) and len(posts) == len(exp["dups"]) and _stops_ok(posts, ref, exp["drug"], exp["dups"])
    return len(posts) == 1 and magnesium_order_ok(posts[0][1], ref, extract_now_from_context(case_data))


def peg_order_ok(pl, ref, now):
    return (order_ok(pl, ref) and mentions(pl, "polyethylene glycol") and dose_is(pl, 17, ("g", "gram"))
            and is_oral(pl) and every_hours(pl, 24) and authored_now(pl, now))


def task3(case_data, results, fhir_api_base):
    exp, posts, ref = ref_task3(case_data, fhir_api_base), extract_posts(results), _ref(case_data)
    if exp["kind"] == "silent":
        return not check_has_post(results)
    if exp["kind"] == "order":
        return len(posts) == 1 and peg_order_ok(posts[0][1], ref, extract_now_from_context(case_data))
    n = len(exp["senna_dups"]) + len(exp["peg_dups"])
    return (len(posts) == n and _stops_ok(posts, ref, SENNA, exp["senna_dups"])
            and _stops_ok(posts, ref, "polyethylene glycol", exp["peg_dups"]))


def heparin_order_ok(pl, ref, now):
    return (order_ok(pl, ref) and mentions(pl, "heparin") and (dose_is(pl, 5000, ("unit",)) or mentions(pl, "5,000") or mentions(pl, "5000"))
            and is_subcutaneous(pl) and every_hours(pl, 8) and authored_now(pl, now))


def task4(case_data, results, fhir_api_base):
    exp, posts = ref_task4(case_data, fhir_api_base), extract_posts(results)
    if exp["kind"] == "silent":
        return not check_has_post(results)
    return len(posts) == 1 and heparin_order_ok(posts[0][1], _ref(case_data), extract_now_from_context(case_data))


def rosuvastatin_order_ok(pl, ref, now):
    return (order_ok(pl, ref) and mentions(pl, "rosuvastatin") and dose_is(pl, 40, ("mg",)) and is_oral(pl)
            and every_hours(pl, 24) and authored_now(pl, now))


def task5(case_data, results, fhir_api_base):
    exp, posts = ref_task5(case_data, fhir_api_base), extract_posts(results)
    if exp["kind"] == "silent":
        return not check_has_post(results)
    return len(posts) == 1 and rosuvastatin_order_ok(posts[0][1], _ref(case_data), extract_now_from_context(case_data))


def metoprolol_order_ok(pl, ref, now):
    return (order_ok(pl, ref) and mentions(pl, "metoprolol succinate") and dose_is(pl, 25, ("mg",)) and is_oral(pl)
            and every_hours(pl, 24) and authored_now(pl, now))


def ecg_order_ok(pl, ref, now):
    return (order_ok(pl, ref, "ServiceRequest") and mentions(pl, "445118002")
            and (now + timedelta(minutes=30)).strftime("%Y-%m-%dT%H:%M") in str(pl.get("occurrenceDateTime", "")))


def task6(case_data, results, fhir_api_base):
    exp, posts, ref = ref_task6(case_data, fhir_api_base), extract_posts(results), _ref(case_data)
    if exp["kind"] == "silent":
        return not check_has_post(results)
    now = extract_now_from_context(case_data)
    return (len(posts) == 2 and sum(metoprolol_order_ok(pl, ref, now) for _, pl in posts) == 1
            and sum(ecg_order_ok(pl, ref, now) for _, pl in posts) == 1)


# ---------------------------------------------------------------- lab-threshold types (task7 - task12)
# The most recent result of one lab within [now - window, now] (inclusive; ties: later effectiveDateTime, then smaller
# id) is compared strictly with the threshold; if it crosses, the listed ServiceRequests are placed (code, priority,
# occurrence = now + offset, quantity where given), otherwise nothing is posted. Each task states its own "now".

LAB = {
    "task7": {"name": "severe anemia", "code": "HGB", "window_h": 24, "op": "<", "thr": 7.0, "orders": [
        {"label": "PRBC transfusion", "code": "18027006", "priority": "stat", "offset_h": 0, "quantity": 1},
        {"label": "repeat hemoglobin", "code": "718-7", "priority": "routine", "offset_h": 4}]},
    "task8": {"name": "hyperkalemia", "code": "K", "window_h": 24, "op": ">", "thr": 5.5, "orders": [
        {"label": "12-lead ECG", "code": "445118002", "priority": "stat", "offset_h": 0},
        {"label": "repeat potassium", "code": "2823-3", "priority": "routine", "offset_h": 2}]},
    "task9": {"name": "hyponatremia", "code": "NA", "window_h": 24, "op": "<", "thr": 130.0, "orders": [
        {"label": "serum osmolality", "code": "2692-2", "priority": "stat", "offset_h": 0},
        {"label": "urine osmolality", "code": "2695-5", "priority": "stat", "offset_h": 0}]},
    "task10": {"name": "elevated ALT", "code": "ALT", "window_h": 168, "op": ">", "thr": 120.0, "orders": [
        {"label": "hepatitis panel", "code": "24363-4", "priority": "routine", "offset_h": 0}]},
    "task11": {"name": "severe hyperglycemia", "code": "GLU", "window_h": 24, "op": ">", "thr": 300.0, "orders": [
        {"label": "beta-hydroxybutyrate", "code": "6873-4", "priority": "stat", "offset_h": 0},
        {"label": "repeat glucose", "code": "2345-7", "priority": "routine", "offset_h": 1}]},
    "task12": {"name": "thrombocytopenia", "code": "PLT", "window_h": 24, "op": "<", "thr": 50.0, "orders": [
        {"label": "hematology referral", "code": "306181000", "priority": "urgent", "offset_h": 0}]},
}


def lab_series(mrn, code, fhir_api_base):
    """(time, value, id) of every pre-cut-off numeric result of `code` for the patient, sorted by time then id."""
    out = []
    for e in _bundle(fhir_api_base, f"Observation?patient={mrn}&code={code}"):
        r = e["resource"]
        if not before_cutoff(r):
            continue
        try:
            out.append((datetime.fromisoformat(r["effectiveDateTime"]), float(r["valueQuantity"]["value"]), str(r.get("id"))))
        except Exception:
            continue
    return sorted(out, key=lambda x: (x[0], x[2]))


def latest_in_window(series, now, window_h):
    w = [x for x in series if now - timedelta(hours=window_h) <= x[0] <= now]
    if not w:
        return None
    tmax = max(x[0] for x in w)
    return min((x for x in w if x[0] == tmax), key=lambda x: x[2])


def crosses(spec, value):
    return value < spec["thr"] if spec["op"] == "<" else value > spec["thr"]


def ref_lab(t, case_data, fhir_api_base):
    spec, now = LAB[t], extract_now_from_context(case_data)
    last = latest_in_window(lab_series(case_data["eval_MRN"], spec["code"], fhir_api_base), now, spec["window_h"])
    kind = "order" if last is not None and crosses(spec, last[1]) else "silent"
    return {"kind": kind, "value": None if last is None else last[1],
            "time": None if last is None else last[0].isoformat(), "now": now.isoformat()}


def lab_order_checks(pl, ref, order, now):
    """(name, passed) for one expected ServiceRequest against one payload (None = no matching POST)."""
    def coded(p):
        return any(str(c.get("code")) == order["code"] for c in ((p.get("code") or {}).get("coding") or []))
    def when(p):
        return (now + timedelta(hours=order["offset_h"])).strftime("%Y-%m-%dT%H:%M") in str(p.get("occurrenceDateTime", ""))
    def qty(p):
        q = p.get("quantityQuantity") or {}
        try:
            return float(q.get("value")) == float(order["quantity"])
        except Exception:
            return False
    items = [(f"{order['label']}: code {order['code']}", coded), (f"{order['label']}: resource / status / intent / patient",
             lambda p: order_ok(p, ref, "ServiceRequest")),
             (f"{order['label']}: priority {order['priority']}", lambda p: str(p.get("priority", "")).lower() == order["priority"]),
             (f"{order['label']}: time now+{order['offset_h']}h", when)]
    if "quantity" in order:
        items.append((f"{order['label']}: quantity {order['quantity']}", qty))
    out = []
    for n, f in items:
        try:
            out.append((n, pl is not None and bool(f(pl))))
        except Exception:
            out.append((n, False))
    return out


def lab_grade(t, case_data, results, fhir_api_base):
    exp = ref_lab(t, case_data, fhir_api_base)
    if exp["kind"] == "silent":
        return not check_has_post(results)
    posts, ref, now = extract_posts(results), _ref(case_data), extract_now_from_context(case_data)
    orders = LAB[t]["orders"]
    if len(posts) != len(orders):
        return False
    return all(sum(all(ok for _, ok in lab_order_checks(pl, ref, o, now)) for _, pl in posts) == 1 for o in orders)


def ref_task7(c, base): return ref_lab("task7", c, base)
def ref_task8(c, base): return ref_lab("task8", c, base)
def ref_task9(c, base): return ref_lab("task9", c, base)
def ref_task10(c, base): return ref_lab("task10", c, base)
def ref_task11(c, base): return ref_lab("task11", c, base)
def ref_task12(c, base): return ref_lab("task12", c, base)


def task7(case_data, results, fhir_api_base): return lab_grade("task7", case_data, results, fhir_api_base)
def task8(case_data, results, fhir_api_base): return lab_grade("task8", case_data, results, fhir_api_base)
def task9(case_data, results, fhir_api_base): return lab_grade("task9", case_data, results, fhir_api_base)
def task10(case_data, results, fhir_api_base): return lab_grade("task10", case_data, results, fhir_api_base)
def task11(case_data, results, fhir_api_base): return lab_grade("task11", case_data, results, fhir_api_base)
def task12(case_data, results, fhir_api_base): return lab_grade("task12", case_data, results, fhir_api_base)
