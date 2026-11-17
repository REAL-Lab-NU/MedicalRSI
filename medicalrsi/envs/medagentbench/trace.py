"""What the Improver is shown of a MedAgentBench episode, read from its trace alone:
  render_reply(text)  an environment reply: a FHIR Bundle becomes one short line per record (type, id, code, time,
                      value / medication / status, last update) instead of raw JSON;
  flags(row)          behaviour problems: no FINISH within the turn limit, an invalid action, the identical action
                      repeated, GET errors, POSTs the server rejected, several actions in one message, a FINISH that
                      claims an order no accepted POST placed."""
from __future__ import annotations

import json
import re

ACTIONS = ("GET", "POST", "FINISH(")
FINISH_CLAIM = re.compile(r"\b(ordered|placed|order(s)? (has|have|was|were) been|referral (has been |was )?(placed|sent|made)|"
                          r"discontinued|stopped|created|scheduled|requested)\b", re.I)
FINISH_DENY = re.compile(r"\b(no (new )?orders?|not (be )?(ordered|placed|needed|required)|no referral|none needed|no action|"
                         r"unable|cannot|could not|no .*(found|available|recorded))\b", re.I)


def _codes(cc):
    if not isinstance(cc, dict):
        return ""
    out = [c.get("code") or c.get("display") for c in cc.get("coding", []) if isinstance(c, dict)]
    out = [x for x in out if x]
    if not out and cc.get("text"):
        out = [cc["text"]]
    return "/".join(str(x) for x in out[:3])


def _record(res):
    t = res.get("resourceType", "?")
    parts = [t, f"id={res.get('id')}"]
    if t == "Observation":
        parts.append(f"code={_codes(res.get('code'))}")
        parts.append(f"time={res.get('effectiveDateTime') or (res.get('effectivePeriod') or {}).get('start')}")
        vq = res.get("valueQuantity")
        if isinstance(vq, dict):
            parts.append(f"value={vq.get('value')} {vq.get('unit') or vq.get('code') or ''}".rstrip())
        elif "valueString" in res:
            parts.append(f"value={res['valueString']!r}")
        interp = [_codes(i) for i in res.get("interpretation", []) if isinstance(i, dict)]
        if any(interp):
            parts.append(f"flag={','.join(x for x in interp if x)}")
    elif t == "MedicationRequest":
        med = res.get("medicationCodeableConcept") or {}
        parts.append(f"med={med.get('text') or _codes(med)!r}")
        parts.append(f"status={res.get('status')}")
        parts.append(f"authoredOn={res.get('authoredOn')}")
        di = (res.get("dosageInstruction") or [{}])[0] if isinstance(res.get("dosageInstruction"), list) else {}
        dr = (di.get("doseAndRate") or [{}])[0] if isinstance(di, dict) and isinstance(di.get("doseAndRate"), list) else {}
        dq = dr.get("doseQuantity") if isinstance(dr, dict) else None
        if isinstance(dq, dict):
            parts.append(f"dose={dq.get('value')} {dq.get('unit') or ''}".rstrip())
        route = di.get("route") if isinstance(di, dict) else None
        if route:
            parts.append(f"route={route.get('text') or _codes(route) if isinstance(route, dict) else route}")
        timing = di.get("timing") if isinstance(di, dict) else None
        if isinstance(timing, dict):
            parts.append(f"timing={(timing.get('code') or {}).get('text') or json.dumps(timing.get('repeat'))}")
    elif t == "Procedure":
        parts.append(f"code={_codes(res.get('code'))}")
        pp = res.get("performedPeriod") or {}
        parts.append(f"performed={res.get('performedDateTime') or pp.get('start')}")
        parts.append(f"status={res.get('status')}")
    elif t == "Condition":
        parts.append(f"code={_codes(res.get('code'))}")
        parts.append(f"clinicalStatus={_codes(res.get('clinicalStatus'))}")
        parts.append(f"onset={res.get('onsetDateTime') or res.get('recordedDate')}")
    elif t == "Patient":
        ids = [i.get("value") for i in res.get("identifier", []) if isinstance(i, dict)]
        name = (res.get("name") or [{}])[0]
        parts.append(f"identifier={','.join(str(i) for i in ids[:2])}")
        parts.append(f"name={' '.join(name.get('given', []))} {name.get('family', '')}".rstrip())
        parts.append(f"birthDate={res.get('birthDate')}")
    elif t == "ServiceRequest":
        parts.append(f"code={_codes(res.get('code'))}")
        parts.append(f"status={res.get('status')}")
        parts.append(f"authoredOn={res.get('authoredOn')}")
    upd = (res.get("meta") or {}).get("lastUpdated")
    if upd:
        parts.append(f"lastUpdated={upd[:10]}")
    return " ".join(p for p in parts if p)


def render_reply(text, max_records=40):
    """One environment reply for the teacher: Bundles as one line per record, everything else as is (clipped)."""
    text = text or ""
    head = "Here is the response from the GET request:"
    if text.startswith(head):
        body = text[len(head):].lstrip()
        try:
            obj, _ = json.JSONDecoder().raw_decode(body)
        except ValueError:
            return text[:1200] + (f" ...[{len(text) - 1200} more chars]" if len(text) > 1200 else "")
        if isinstance(obj, dict) and obj.get("resourceType") == "Bundle":
            entries = [e.get("resource", {}) for e in obj.get("entry", []) if isinstance(e, dict)]
            nxt = any(l.get("relation") == "next" for l in obj.get("link", []) if isinstance(l, dict))
            lines = [f"GET response: Bundle with {len(entries)} records on this page"
                     + (f" (total {obj['total']})" if "total" in obj else "") + ("; a NEXT page link is present" if nxt else "")]
            lines += [f"  {i + 1}. {_record(r)}" for i, r in enumerate(entries[:max_records])]
            if len(entries) > max_records:
                lines.append(f"  ... {len(entries) - max_records} more records")
            return "\n".join(lines)
        return "GET response (non-Bundle): " + json.dumps(obj)[:1200]
    return text[:1200] + (f" ...[{len(text) - 1200} more chars]" if len(text) > 1200 else "")


def _agent_msgs(row):
    return [(i, m["content"] or "") for i, m in enumerate(row["history"]) if m["role"] == "agent"]


def _accepted_posts(row):
    """[(index, url, payload or None)] for the student's POSTs the server accepted."""
    h, out = row["history"], []
    for i, m in enumerate(h):
        c = (m["content"] or "").lstrip()
        if m["role"] == "agent" and c.startswith("POST") and i + 1 < len(h) and (h[i + 1]["content"] or "").startswith("POST request accepted"):
            url = c.split("\n")[0][4:].strip()
            try:
                payload = json.loads(c.split("\n", 1)[1])
            except Exception:
                payload = None
            out.append((i, url, payload))
    return out


def flags(row):
    h, out = row["history"], []
    status = row.get("status")
    if status == "task limit reached":
        out.append("Used all its turns without ever calling FINISH, so nothing it did was graded as an answer.")
    elif status == "agent invalid action":
        bad = _agent_msgs(row)[-1][1] if _agent_msgs(row) else ""
        out.append(f"Its last message was not a valid action, which ended the episode: {bad[:220]!r}. Each turn must be exactly "
                   "one plain `GET url`, `POST url` + JSON, or `FINISH([...])`.")
    elif status == "agent context limit":
        out.append("Ran out of context (the conversation became too long).")
    msgs = _agent_msgs(row)
    run, prev = 1, None
    for (i, c) in msgs:
        if c == prev:
            run += 1
        else:
            if run >= 2:
                out.append(f"Sent the identical action {run} times in a row ({prev[:150]!r}) instead of changing it after the reply.")
            run, prev = 1, c
    if run >= 2:
        out.append(f"Sent the identical action {run} times in a row ({prev[:150]!r}) instead of changing it after the reply.")
    errs = [(h[i - 1]["content"] or "")[:160] for i, m in enumerate(h) if m["role"] != "agent" and i > 0
            and (m["content"] or "").startswith("Error in sending the GET request")]
    if errs:
        out.append(f"{len(errs)} GET request(s) failed with an HTTP error, e.g. {errs[0]!r}.")
    rej = sum(1 for m in h if m["role"] != "agent" and (m["content"] or "").startswith("Invalid POST request"))
    if rej:
        out.append(f"The server rejected {rej} POST(s) as invalid: the text after the POST line must be exactly one JSON "
                   "object - nothing may follow it in the same message (no second POST, no FINISH).")
    multi = [c for _, c in msgs if sum(1 for line in c.split("\n") if line.strip().startswith(ACTIONS)) >= 2]
    if multi:
        out.append(f"Put several actions into one message {len(multi)} time(s); only one action per turn is executed.")
    fin = [c for _, c in msgs if c.lstrip().startswith("FINISH(")]
    if fin and not _accepted_posts(row) and FINISH_CLAIM.search(fin[-1]) and not FINISH_DENY.search(fin[-1]):
        out.append("Its FINISH says an order/action was carried out, but no POST was accepted in this episode: writing "
                   "'ordered' in FINISH places nothing.")
    return out
