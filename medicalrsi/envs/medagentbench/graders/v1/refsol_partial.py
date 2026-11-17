"""Process score for MedAgentBench v1. The reference grader (refsol.taskN) decides the outcome; a failed attempt gets
0.5 x the mean over its task's check groups of the share of passed checks:
  query   the agent fetched the data the reference solution reads;
  action  the POST decision, every condition the reference grader asserts on the POSTs, and "called FINISH";
  answer  the FINISH value, where the reference grader checks it.
All checks passing implies that the reference grader passes."""
import json
from datetime import datetime, timedelta
from urllib.parse import parse_qs, urlparse

from .utils import send_get_request

NOW = datetime.fromisoformat("2023-11-13T10:15:00+00:00")


def _role(m):
    return m.role if hasattr(m, "role") else m.get("role")


def _content(m):
    return (m.content if hasattr(m, "content") else m.get("content")) or ""


def _agent(results):
    return [_content(m) for m in results.history if _role(m) == "agent"]


def _has_post(results):
    """Same rule as refsol.check_has_post: any agent message containing 'POST'."""
    return any("POST" in c for c in _agent(results))


def _posts(results):
    """Same parsing as refsol.extract_posts: accepted POSTs as (url, payload)."""
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


def _fetched(results, resource, mrn=None, code=None):
    """The agent sent a GET for `resource` of this patient (patient/subject/identifier = MRN) with this code."""
    for c in _agent(results):
        c = c.strip()
        if not c.startswith("GET"):
            continue
        u = urlparse(c[3:].strip())
        if not u.path.rstrip("/").endswith("/" + resource):
            continue
        q = {k.lower(): ",".join(v) for k, v in parse_qs(u.query).items()}
        if mrn is not None:
            who = " ".join(q.get(k, "") for k in ("patient", "subject", "identifier", "_id"))
            if mrn not in who:
                continue
        if code is not None and code.lower() not in [x.strip().lower() for x in q.get("code", "").split(",")]:
            continue
        return True
    return False


def _answer(results):
    try:
        return json.loads(results.result)
    except Exception:
        return "<unparseable>"


def _ok(fn):
    try:
        return bool(fn())
    except Exception:
        return False


def _first_post(posts, endpoint):
    for url, payload in posts:
        if url.rstrip("/").endswith("/" + endpoint):
            return url, payload
    return None, None


def _obs(base, mrn, code):
    url = f"{base}Observation?patient={mrn}&code={code}&_count=5000&_format=json"
    return json.loads(send_get_request(url)["data"]).get("entry", [])


def _last(entries, within_24h=False):
    last_t, last_v, last_s = None, None, None
    for e in entries:
        t = datetime.fromisoformat(e["resource"]["effectiveDateTime"])
        if within_24h and t < NOW - timedelta(hours=24):
            continue
        if last_t is None or t > last_t:
            last_t, last_v, last_s = t, e["resource"]["valueQuantity"]["value"], e["resource"]["effectiveDateTime"]
    return last_t, last_v, last_s


def _fields(payload, checks):
    return [(name, payload is not None and _ok(lambda f=f: f(payload))) for name, f in checks]


def _subject(mrn):
    return ("subject", lambda p: p["subject"] == {"reference": f"Patient/{mrn}"})


def t1(c, r, base):
    return {"query": [("GET Patient", _fetched(r, "Patient"))],
            "action": [("no POST", not _has_post(r))],
            "answer": [(f"answer = MRN (expected {json.dumps(c['sol'])})", _answer(r) == c["sol"])]}


def t2(c, r, base):
    mrn = c["eval_MRN"]
    get = json.loads(send_get_request(f"{base}Patient?identifier={mrn}&_format=json")["data"])
    dob = datetime.strptime(get["entry"][0]["resource"]["birthDate"], "%Y-%m-%d")
    age = NOW.year - dob.year - ((NOW.month, NOW.day) < (dob.month, dob.day))
    return {"query": [("GET Patient of MRN", _fetched(r, "Patient", mrn))],
            "action": [("no POST", not _has_post(r))],
            "answer": [(f"answer = [age] (expected [{age}])", _answer(r) == [age])]}


def t3(c, r, base):
    mrn, posts = c["eval_MRN"], _posts(r)
    url, p = _first_post(posts, "Observation")
    return {"action": [("exactly 1 POST", len(posts) == 1),
                       ("first POST to Observation", bool(posts) and posts[0][0] == f"{base}Observation")]
            + _fields(p, [
                ("resourceType", lambda p: p["resourceType"] == "Observation"),
                ("category vital-signs", lambda p: len(p["category"]) == 1 and len(p["category"][0]["coding"]) == 1
                 and p["category"][0]["coding"][0] == {"system": "http://hl7.org/fhir/observation-category",
                                                       "code": "vital-signs", "display": "Vital Signs"}),
                ("code BP", lambda p: p["code"] == {"text": "BP"}),
                ("effectiveDateTime", lambda p: p["effectiveDateTime"] == "2023-11-13T10:15:00+00:00"),
                ("status final", lambda p: p["status"] == "final"),
                ("valueString", lambda p: p["valueString"] == "118/77 mmHg"),
                _subject(mrn)])}


def t4(c, r, base):
    mrn = c["eval_MRN"]
    _, v, _ = _last(_obs(base, mrn, "MG"), within_24h=True)
    return {"query": [("GET Observation MG", _fetched(r, "Observation", mrn, "MG"))],
            "action": [("no POST", not _has_post(r))],
            "answer": [(f"answer = last MG in 24h (expected {json.dumps([v if v is not None else -1])})", _answer(r) == [v if v is not None else -1])]}


def t5(c, r, base):
    mrn = c["eval_MRN"]
    _, v, _ = _last(_obs(base, mrn, "MG"), within_24h=True)
    g = {"query": [("GET Observation MG", _fetched(r, "Observation", mrn, "MG"))]}
    if v is None:                                   # original: no POST is all that is checked
        g["action"] = [("no POST (no recent MG)", not _has_post(r))]
        return g
    a = _answer(r)
    g["answer"] = [(f"answer = last MG or [] (expected {json.dumps([v])} or [])", a == [v] or a == [])]
    if v > 1.9:
        g["action"] = [("no POST (MG not low)", not _has_post(r))]
        return g
    dose = 4 if v < 1 else 2 if v < 1.5 else 1
    posts = _posts(r)
    _, p = _first_post(posts, "MedicationRequest")
    g["action"] = [("exactly 1 POST", len(posts) == 1),
                   ("first POST to MedicationRequest", bool(posts) and posts[0][0] == f"{base}MedicationRequest")] \
        + _fields(p, [
            ("resourceType", lambda p: p["resourceType"] == "MedicationRequest"),
            ("NDC system", lambda p: p["medicationCodeableConcept"]["coding"][0]["system"] == "http://hl7.org/fhir/sid/ndc"),
            ("NDC code", lambda p: p["medicationCodeableConcept"]["coding"][0]["code"] == "0338-1715-40"),
            ("authoredOn", lambda p: "2023-11-13T10:15" in p["authoredOn"]),
            ("route IV", lambda p: p["dosageInstruction"][0]["route"] == "IV"),
            (f"dose {dose} g", lambda p: p["dosageInstruction"][0]["doseAndRate"][0]["doseQuantity"] == {"value": dose, "unit": "g"}),
            (f"rate {dose} h", lambda p: p["dosageInstruction"][0]["doseAndRate"][0]["rateQuantity"] == {"value": dose, "unit": "h"}),
            ("status active", lambda p: p["status"] == "active"),
            ("intent order", lambda p: p["intent"] == "order"),
            _subject(mrn)])
    return g


def t6(c, r, base):
    mrn = c["eval_MRN"]
    vals = [e["resource"]["valueQuantity"]["value"] for e in _obs(base, mrn, "GLU")
            if datetime.fromisoformat(e["resource"]["effectiveDateTime"]) >= NOW - timedelta(hours=24)]
    ref = sum(vals) / len(vals) if vals else -1
    a = _answer(r)
    return {"query": [("GET Observation GLU", _fetched(r, "Observation", mrn, "GLU"))],
            "action": [("no POST", not _has_post(r))],
            "answer": [(f"answer = 24h mean GLU (±0.1) (expected [{round(ref, 2)}])", _ok(lambda: len(a) == 1 and abs(a[0] - ref) < 0.1))]}


def t7(c, r, base):
    mrn = c["eval_MRN"]
    _, v, _ = _last(_obs(base, mrn, "GLU"))
    return {"query": [("GET Observation GLU", _fetched(r, "Observation", mrn, "GLU"))],
            "action": [("no POST", not _has_post(r))],
            "answer": [(f"answer = last GLU (expected {json.dumps([v if v is not None else -1])})", _answer(r) == [v if v is not None else -1])]}


def t8(c, r, base):
    mrn, posts = c["eval_MRN"], _posts(r)
    _, p = _first_post(posts, "ServiceRequest")
    note = ("Situation: acute left knee injury, Background: radiology report indicates ACL tear. Assessment: ACL tear "
            "grade II. Recommendation: request for Orthopedic service to evaluate and provide management recommendations.")
    return {"action": [("exactly 1 POST", len(posts) == 1),
                       ("first POST to ServiceRequest", bool(posts) and posts[0][0] == f"{base}ServiceRequest")]
            + _fields(p, [
                ("resourceType", lambda p: p["resourceType"] == "ServiceRequest"),
                ("SNOMED system", lambda p: p["code"]["coding"][0]["system"] == "http://snomed.info/sct"),
                ("SNOMED code", lambda p: p["code"]["coding"][0]["code"] == "306181000000106"),
                ("authoredOn", lambda p: p["authoredOn"] == "2023-11-13T10:15:00+00:00"),
                ("status active", lambda p: p["status"] == "active"),
                ("intent order", lambda p: p["intent"] == "order"),
                ("priority stat", lambda p: p["priority"] == "stat"),
                ("SBAR note", lambda p: note in p["note"]["text"]),
                _subject(mrn)])}


def t9(c, r, base):
    mrn = c["eval_MRN"]
    _, v, _ = _last(_obs(base, mrn, "K"))
    a = _answer(r)
    g = {"query": [("GET Observation K", _fetched(r, "Observation", mrn, "K"))],
         "answer": [(f"answer = last K or [] (expected {json.dumps([v if v is not None else -1])} or [])", a == [v if v is not None else -1] or a == [])]}
    if v is None or v >= 3.5:
        g["action"] = [("no POST (K not low)", not _has_post(r))]
        return g
    dose = (3.5 - v) / 0.1 * 10
    posts = _posts(r)
    _, m = _first_post(posts, "MedicationRequest")
    _, s = _first_post(posts, "ServiceRequest")

    def route_ok(p):
        route = p["dosageInstruction"][0]["route"]
        rs = route.get("text", route.get("coding", [{}])[0].get("display", "")) if isinstance(route, dict) else route
        return rs.lower().strip() == "oral"
    g["action"] = [("exactly 2 POSTs", len(posts) == 2),
                   ("POST order MedicationRequest then ServiceRequest",
                    len(posts) >= 2 and posts[0][0] == f"{base}MedicationRequest" and posts[1][0] == f"{base}ServiceRequest")] \
        + _fields(m, [
            ("K: resourceType", lambda p: p["resourceType"] == "MedicationRequest"),
            ("K: NDC system", lambda p: p["medicationCodeableConcept"]["coding"][0]["system"] == "http://hl7.org/fhir/sid/ndc"),
            ("K: NDC code", lambda p: p["medicationCodeableConcept"]["coding"][0]["code"] == "40032-917-01"),
            ("K: authoredOn", lambda p: "2023-11-13T10:15" in p["authoredOn"]),
            ("K: route oral", route_ok),
            (f"K: dose {round(dose, 1)} mEq", lambda p: abs(p["dosageInstruction"][0]["doseAndRate"][0]["doseQuantity"]["value"] - dose) <= 0.1),
            ("K: unit mEq", lambda p: p["dosageInstruction"][0]["doseAndRate"][0]["doseQuantity"]["unit"] == "mEq"),
            ("K: status active", lambda p: p["status"] == "active"),
            ("K: intent order", lambda p: p["intent"] == "order"),
            ("K: subject", lambda p: p["subject"] == {"reference": f"Patient/{mrn}"})]) \
        + _fields(s, [
            ("lab: resourceType", lambda p: p["resourceType"] == "ServiceRequest"),
            ("lab: LOINC system", lambda p: p["code"]["coding"][0]["system"] == "http://loinc.org"),
            ("lab: LOINC code", lambda p: p["code"]["coding"][0]["code"] == "2823-3"),
            ("lab: authoredOn", lambda p: p["authoredOn"] == "2023-11-13T10:15:00+00:00"),
            ("lab: status active", lambda p: p["status"] == "active"),
            ("lab: intent order", lambda p: p["intent"] == "order"),
            ("lab: priority stat", lambda p: p["priority"] == "stat"),
            ("lab: subject", lambda p: p["subject"] == {"reference": f"Patient/{mrn}"}),
            ("lab: next morning 08:00", lambda p: "2023-11-14T08:" in p["occurrenceDateTime"])])
    return g


def t10(c, r, base):
    mrn = c["eval_MRN"]
    t, v, s = _last(_obs(base, mrn, "A1C"))
    ref = [-1] if v is None else [v, s]
    a = _answer(r)
    g = {"query": [("GET Observation A1C", _fetched(r, "Observation", mrn, "A1C"))],
         "answer": [(f"answer = last A1C [value, time] or [] (expected {json.dumps(ref)} or [])", a == ref or a == [])]}
    if v is not None and t >= datetime.fromisoformat("2022-11-13T10:15:00+00:00"):
        g["action"] = [("no POST (A1C recent)", not _has_post(r))]
        return g
    posts = _posts(r)
    _, p = _first_post(posts, "ServiceRequest")
    g["action"] = [("exactly 1 POST", len(posts) == 1),
                   ("first POST to ServiceRequest", bool(posts) and posts[0][0] == f"{base}ServiceRequest")] \
        + _fields(p, [
            ("resourceType", lambda p: p["resourceType"] == "ServiceRequest"),
            ("LOINC system", lambda p: p["code"]["coding"][0]["system"] == "http://loinc.org"),
            ("LOINC code", lambda p: p["code"]["coding"][0]["code"] == "4548-4"),
            ("authoredOn", lambda p: p["authoredOn"] == "2023-11-13T10:15:00+00:00"),
            ("status active", lambda p: p["status"] == "active"),
            ("intent order", lambda p: p["intent"] == "order"),
            ("priority stat", lambda p: p["priority"] == "stat"),
            _subject(mrn)])
    return g


CHECKS = {f"task{i}": f for i, f in enumerate((t1, t2, t3, t4, t5, t6, t7, t8, t9, t10), 1)}


def score(case_data, results, fhir_api_base, full: bool):
    """{'score', 'full', 'frac', 'groups': {group: [[check, passed], ...]}}; results is the episode's TaskOutput
    (history + FINISH result), full the original grader's verdict."""
    groups = CHECKS[case_data["id"].split("_")[0]](case_data, results, fhir_api_base)
    groups.setdefault("action", []).append(("called FINISH", results.result is not None))   # no FINISH, no grading
    fracs = [sum(ok for _, ok in g) / len(g) for g in groups.values() if g]
    frac = sum(fracs) / len(fracs) if fracs else 0.0
    return {"score": 1.0 if full else round(0.5 * frac, 4), "full": bool(full), "frac": round(frac, 4),
            "groups": {k: [[n, bool(ok)] for n, ok in g] for k, g in groups.items()}}
