"""An offline FHIR server for the MedAgentBench data: serves a snapshot of the MedAgentBench HAPI FHIR server
(fhir_snapshot.py) and answers every GET as HAPI FHIR 7.6.0 does (same bytes, same search semantics, same errors).

    python -m medicalrsi.envs.medagentbench.fhir_server --snapshot data/medagentbench/fhir_snapshot --port 8080
"""
from __future__ import annotations

import argparse
import gzip
import json
import re
import sys
import threading
import time
import unicodedata
import urllib.parse
import uuid
from collections import OrderedDict, defaultdict
from datetime import timezone
from decimal import Decimal
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from medicalrsi.envs.medagentbench.hapi_hash import escape, hapi_hash
from medicalrsi.envs.medagentbench.hapi_json import dumps_compact, dumps_pretty, dumps_xml, loads

DATA_TYPES = ("Patient", "Observation", "Condition", "MedicationRequest", "Procedure")
DEFAULT_COUNT = 20
PREFETCH = 13            # HAPI search pre-fetch threshold: fewer matches -> the search is complete on the first page
MAX_STRING = 768         # longest string search value HAPI accepts (HAPI-1260)
MAX_HEADER = 8192        # Jetty's request header buffer: a longer request line + headers is answered 400 before HAPI
CONTENT_TYPE = "application/fhir+json;charset=UTF-8"
NS = uuid.UUID("6f1d3a52-6a43-4f8e-9a77-0c9c3f0b8e11")
IGNORED_SPECIALS = {"_since", "_until", "_lastN", "_securityContext", "_pretty", "_format", "_getpages",
                    "_getpagesoffset", "_bundletype", "_count", "_sort", "_summary", "_total", "_elements",
                    "_include", "_revinclude", "_include:iterate", "_revinclude:iterate", "_include:recurse",
                    "_revinclude:recurse", "_contained", "_containedType", "_offset"}
FULLTEXT = {"_text", "_content"}
UTC = timezone.utc
NEG_INF, POS_INF = -(1 << 62), (1 << 62)


class Bad(Exception):
    def __init__(self, status, code, msg):
        super().__init__(msg)
        self.status, self.code, self.msg = status, code, msg


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------------------------- value helpers

def norm_str(s):
    """HAPI StringUtil.normalizeStringForSearchIndexing: strip accents, upper case."""
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn").upper()


# --------------------------------------------------- search value parsing, ported from HAPI 7.6 (same errors)
PREFIXES = {"eq", "ne", "gt", "lt", "ge", "le", "sa", "eb", "ap"}
LEGACY_PREFIXES = {">=": "ge", ">": "gt", "<=": "le", "<": "lt", "~": "ap", "=": "eq"}


def hapi_prefix(v):
    """BaseParamWithPrefix.extractPrefixAndReturnRest: the prefix is everything before the first '-', '%' or digit."""
    off = 0
    while off < len(v) and not (v[off] in "-%" or v[off].isdecimal()):
        off += 1
    if off > 0 and off == len(v):
        raise Bad(400, "HAPI-1940", f'Invalid date/time/quantity format: "{v}"')
    p = v[:off]
    if p.strip() == "":
        return None, v[off:]
    if p in PREFIXES:
        return p, v[off:]
    if p in LEGACY_PREFIXES:
        return LEGACY_PREFIXES[p], v[off:]
    raise Bad(400, "HAPI-1941", f'Invalid prefix: "{p}"')


def _bad_date(v, detail=None):
    if detail:
        raise Bad(400, "HAPI-1883", f'Invalid date/time format: "{v}": {detail}')
    raise Bad(400, "HAPI-1882", f'Invalid date/time format: "{v}"')


def _jint(v, s, lo, hi):
    """Integer.parseInt plus HAPI's range check"""
    if not re.fullmatch(r"[+-]?\d+", s):
        _bad_date(v)
    n = int(s)
    if n < lo or n > hi:
        _bad_date(v)
    return n


def _expect(v, i, ch):
    if v[i] != ch:
        _bad_date(v, f"Expected character '{ch}' at index {i} but found {v[i]}")


def _offset_index(v):
    p, m, z = v.find("+", 16), v.find("-", 16), v.find("Z", 16)
    r = max(p, m, z)
    if r == -1:
        return -1
    if r - 2 != p + m + z:
        _bad_date(v)
    return r


def _tz_minutes(v, s):
    if s[0] == "Z":
        return 0
    if len(s) != 6 or s[3] != ":" or s[0] not in "+-":
        _bad_date(v, 'Timezone offset must be in the form "Z", "-HH:mm", or "+HH:mm"')
    return (1 if s[0] == "+" else -1) * (_jint(v, s[1:3], 0, 23) * 60 + _jint(v, s[4:6], 0, 59))


def _days_from_civil(y, m, d):
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    return era * 146097 + yoe * 365 + yoe // 4 - yoe // 100 + doy - 719468


def hapi_date(value):
    """BaseDateTimeType.parse of a search value -> (low_ms, high_ms inclusive, first day, last day as yyyymmdd,
    has a time), None if blank; raises 400 exactly where HAPI does (minute precision is accepted, an hour alone is
    not; '...Z<anything>' is accepted; offsets need 'HH:mm'). No offset = the server's zone (UTC)."""
    v = value.strip() if value and (value[0] == " " or value[-1] == " ") else value
    n = len(v)
    if n == 0:
        return None
    if n < 4:
        _bad_date(v)
    y, mo, d, hh, mi, ss, ms, tz, unit = _jint(v, v[0:4], 0, 9999), 1, 1, 0, 0, 0, 0, 0, "year"
    if n > 4:
        _expect(v, 4, "-")
        if n < 7:
            _bad_date(v)
        mo, unit = _jint(v, v[5:7], 1, 12), "month"
        if n > 7:
            _expect(v, 7, "-")
            if n < 10:
                _bad_date(v)
            leap = y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)
            mdays = [31, 29 if leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][mo - 1]
            d, unit = _jint(v, v[8:10], 1, mdays), "day"
            if n > 10:
                if n < 16:                          # yyyy-mm-ddThh:mm (minute precision is accepted)
                    _bad_date(v)
                _expect(v, 10, "T")
                off = _offset_index(v)
                if off == -1:
                    time_s = v[11:]
                else:
                    time_s = v[11:off]
                    tz = _tz_minutes(v, v[off:])
                _expect(v, 13, ":")
                hh, mi, unit = _jint(v, v[11:13], 0, 23), _jint(v, v[14:16], 0, 59), "minute"
                if len(time_s) > 5:
                    if n < 19:
                        _bad_date(v)
                    _expect(v, 16, ":")
                    ss, unit = _jint(v, v[17:19], 0, 59), "second"
                    if len(time_s) > 8:
                        _expect(v, 19, ".")
                        if n < 20:
                            _bad_date(v)
                        end = _offset_index(v)
                        end = n if end == -1 else end
                        frac = v[20:min(end, 23)] if end > 23 else v[20:end]
                        ms = _jint(v, frac, 0, 999) * {1: 100, 2: 10}.get(len(frac), 1)
                        unit = "milli"
    day = _days_from_civil(y, mo, d)
    lo = ((day * 24 + hh) * 60 + mi - tz) * 60000 + ss * 1000 + ms
    first = last = y * 10000 + mo * 100 + d
    if unit == "year":
        hi = (_days_from_civil(y + 1, 1, 1) - day) * 86400000 + lo
        last = y * 10000 + 1231
    elif unit == "month":
        hi = (_days_from_civil(y + (mo == 12), mo % 12 + 1, 1) - day) * 86400000 + lo
        last = y * 10000 + mo * 100 + (_days_from_civil(y + (mo == 12), mo % 12 + 1, 1) - day)
    else:
        hi = lo + {"day": 86400000, "minute": 60000, "second": 1000, "milli": 1}[unit]
    return lo, hi - 1, first, last, unit not in ("year", "month", "day")


def index_date(text):
    """HAPI's date index row for a stored date / dateTime / instant -> (low_ms, high_ms, low_day, high_day): a value
    with a time is one instant (low = high) on the calendar day written in the string; a date (yyyy / yyyy-mm /
    yyyy-mm-dd) runs from its first day 00:00 to its last day 23:59:59.999 (server zone UTC)."""
    try:
        r = hapi_date(text) if isinstance(text, str) else None
    except Bad:
        return None
    if r is None:
        return None
    lo, hi, first, last, timed = r
    if timed:
        d = int(text[:10].replace("-", ""))
        return lo, lo, d, d
    return lo, hi, first, last


def split_or(v):
    """QualifiedParamList: OR values are separated by unescaped commas"""
    return [x.replace("\\,", ",") for x in re.split(r"(?<!\\),", v)]


def date_range_param(items):
    """DateRangeParam.setValuesAsQueryTokens + addParam over every value of one date parameter (any modifier) ->
    [(prefix, hapi_date(value))]: no OR lists; at most one lower bound (gt ge sa) and one upper bound (lt le eb); eq / ne / a
    blank value / :missing take both; 'ap' is refused. Each bound is then matched on its own (date_match)."""
    lower = upper = False
    out = []
    for mod, raw in items:
        if len(split_or(raw)) > 1:
            raise Bad(400, "HAPI-1930", "DateRange parameter does not support OR queries")
        prefix, rng = None, None
        v = raw.replace(" ", "+")                  # a '+' the URL decoding turned into a space
        if mod != "missing" and v.strip():
            prefix, rest = hapi_prefix(v)
            rng = hapi_date(rest)
        p = prefix or "eq"
        if p in ("eq", "ne"):
            if lower or upper:
                raise Bad(400, "HAPI-1922", "Can not have multiple date range parameters for the same param without a qualifier")
            lower = upper = True
        elif p in ("gt", "ge", "sa"):
            if lower:
                raise Bad(400, "HAPI-1923", "Can not have multiple date range parameters for the same param that specify a lower bound")
            lower = True
        elif p in ("lt", "le", "eb"):
            if upper:
                raise Bad(400, "HAPI-1924", "Can not have multiple date range parameters for the same param that specify an upper bound")
            upper = True
        else:
            raise Bad(400, "HAPI-1925", "Unknown comparator: APPROXIMATE")
        if rng is not None:
            out.append((p, rng))
    return out


def split_parameter_string(v, delim="|"):
    """ParameterUtil.splitParameterString (a delimiter at index 0 is kept as text; empty parts are None)"""
    out, b = [], []
    for i, c in enumerate(v):
        if c == delim and i > 0 and v[i - 1] != "\\":
            out.append("".join(b) or None)
            b = []
        else:
            b.append(c)
    if b:
        out.append("".join(b))
    return out


_BIGDECIMAL = re.compile(r"[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?")


def quantity_param(a):
    """QuantityParam: [prefix]number|system|code -> (prefix, Decimal | None, system, code). HAPI answers a number
    BigDecimal cannot read with a 500; 'sa' / 'eb' with 400 HAPI-1235."""
    parts = split_parameter_string(a)
    prefix, num = None, None
    if parts and parts[0] and parts[0].strip():
        prefix, rest = hapi_prefix(parts[0])
        if not _BIGDECIMAL.fullmatch(rest):
            raise Bad(500, "HAPI-0000", f"Character array is missing \"exponent\" mark 'e' or 'E'. ({rest})")
        num = Decimal(rest)
        if prefix in ("sa", "eb"):
            raise Bad(400, "HAPI-1235", f'Unable to handle quantity prefix "{prefix}" for value: {a}')
    system = parts[1] if len(parts) > 1 and parts[1] and parts[1].strip() else None
    code = parts[2] if len(parts) > 2 and parts[2] and parts[2].strip() else None
    return prefix or "eq", num, system, code


def iddt(v):
    """IdDt.setValue -> (base_url, resource_type, id_part); the version (/_history/n) is dropped."""
    if v.startswith("#") and len(v) > 1 or v.startswith("urn:"):
        return None, None, v
    vi = v.find("/_history/")
    if vi != -1:
        idx = v.rfind("/", 0, vi)
        idp = v[idx + 1:vi]
    else:
        idx = v.rfind("/")
        idp = v[idx + 1:]
    base = rtype = None
    if idx > 0:
        ti = v.rfind("/", 0, idx)
        if ti == -1:
            rtype = v[:idx]
        else:
            if ti > 0 and v[ti - 1] == "/":
                ti = v.find("/", ti + 1)
            if ti >= idx:
                rtype, idp = idp, None
            else:
                rtype = v[ti + 1:idx]
            if ti > 4:
                base = v[:ti]
    return base, rtype, idp


REF_MODIFIERS = {"missing", "contains", "exact", "in", "iterate", "not-in", "of-type", "recurse", "text"}


def date_match(prefix, s, t):
    """one date bound s = hapi_date(...) against one index row t = index_date(...), as HAPI 7.6 builds the SQL
    (measured on a 360-query grid). A search value of day precision or coarser is
    compared on calendar days, one with a time on instants. HAPI quirk reproduced: 'ge' with a year / month starts
    at its LAST day (date=ge2023 finds nothing dated 2023 before Dec 31) and 'le' with a year / month ends at its
    FIRST day; 'gt' / 'sa' / 'lt' / 'eb' / 'eq' / 'ne' follow the FHIR definitions."""
    slo, shi, first, last, timed = s
    if timed:
        a, b, lo, hi = slo, shi, t[0], t[1]
        if prefix == "ge":
            return hi >= a
        if prefix == "le":
            return lo <= b
    else:
        a, b, lo, hi = first, last, t[2], t[3]
        if prefix == "ge":
            return hi >= b
        if prefix == "le":
            return lo <= a
    if prefix == "eq":
        return lo >= a and hi <= b
    if prefix == "ne":
        return lo < a or hi > b
    if prefix == "gt":
        return hi > b
    if prefix == "sa":
        return lo > b
    if prefix == "lt":
        return lo < a
    if prefix == "eb":
        return hi < a
    return False


AP_FUZZ = Decimal(0.1)        # new BigDecimal(0.1): the double's exact value, as HAPI computes the 'ap' window


def num_match(prefix, v, x):
    """quantity / number (NumberPredicateBuilder): eq / ne are exact numeric comparisons (HAPI dropped the implicit
    precision range in 2019); ap = within v * 0.1 (an empty window for negative v, as in HAPI)."""
    if x is None:
        return False
    x = Decimal(str(x))
    if prefix == "eq":
        return x == v
    if prefix == "ne":
        return x != v
    if prefix == "gt":
        return x > v
    if prefix == "lt":
        return x < v
    if prefix == "ge":
        return x >= v
    if prefix == "le":
        return x <= v
    if prefix == "ap":
        mul = v * AP_FUZZ
        return v - mul <= x <= v + mul
    return False


def codings(cc):
    """CodeableConcept / Coding / list of them -> [(system, code)]."""
    out = []
    for c in cc if isinstance(cc, list) else [cc]:
        if not isinstance(c, dict):
            continue
        if "coding" in c:
            out += [(x.get("system"), x.get("code")) for x in c.get("coding") or []]
        elif "code" in c or "system" in c:
            out.append((c.get("system"), c.get("code")))
    return out


def cc_texts(cc):
    out = []
    for c in cc if isinstance(cc, list) else [cc]:
        if isinstance(c, dict):
            if c.get("text"):
                out.append(c["text"])
            out += [x.get("display") for x in c.get("coding") or [] if x.get("display")]
    return out


def period_or_dt(r, base):
    """choice element <base>[x]: dateTime / instant / date / Period -> index rows"""
    out = []
    for k in (base + "DateTime", base + "Instant", base + "Date"):
        rg = index_date(r.get(k))
        if rg:
            out.append(rg)
    p = r.get(base + "Period")
    if isinstance(p, dict):
        lo, hi = index_date(p.get("start")), index_date(p.get("end"))
        out.append((lo[0] if lo else NEG_INF, hi[1] if hi else POS_INF, lo[2] if lo else 0, hi[3] if hi else 99999999))
    return out


def _dates(*values):
    return [rg for v in values for rg in [index_date(v)] if rg]


def ref_values(x):
    out = []
    for v in x if isinstance(x, list) else [x]:
        if isinstance(v, dict) and isinstance(v.get("reference"), str):
            out.append(v["reference"])
    return out


# ------------------------------------------------------------------- search parameter definitions (data types)
# kind, extractor(resource) -> values.  token: [(system, code)]; string: [str]; date: [(lo, hi)]; reference:
# ["Type/id"]; quantity: [(value, system, code, unit)]; uri: [str].

def _subject(r):
    return ref_values(r.get("subject"))


def _patient_ref(r):
    return [v for v in ref_values(r.get("subject")) if v.startswith("Patient/")]


def _ident(r):
    return [(i.get("system"), i.get("value")) for i in r.get("identifier") or []]


def _q(q):
    return [(q.get("value"), q.get("system"), q.get("code"), q.get("unit"))] if isinstance(q, dict) else []


def _status(r):
    return [(None, r["status"])] if isinstance(r.get("status"), str) else []


def _names(r, part):
    out = []
    for n in r.get("name") or []:
        if part in ("family", "name") and n.get("family"):
            out.append(n["family"])
        if part in ("given", "name"):
            out += n.get("given") or []
        if part == "name":
            out += (n.get("prefix") or []) + (n.get("suffix") or [])
            if n.get("text"):
                out.append(n["text"])
    return out


def _addr(r, field):
    out = []
    for a in r.get("address") or []:
        if field == "address":
            out += a.get("line") or []
            out += [a[k] for k in ("city", "district", "state", "postalCode", "country", "text") if a.get(k)]
        elif a.get(field):
            out.append(a[field])
    return out


def _telecom(r, system=None):
    return [(None, t.get("value")) for t in r.get("telecom") or [] if t.get("value") and (system is None or t.get("system") == system)]


COMMON = {
    "_id": ("token", lambda r: [(None, r.get("id"))]),
    "_lastUpdated": ("date", lambda r: _dates((r.get("meta") or {}).get("lastUpdated"))),
    "_security": ("token", lambda r: codings(r.get("meta", {}).get("security") or [])),
    "_tag": ("token", lambda r: codings(r.get("meta", {}).get("tag") or [])),
    "_profile": ("uri", lambda r: list(r.get("meta", {}).get("profile") or [])),
    "_source": ("uri", lambda r: [r["meta"]["source"]] if r.get("meta", {}).get("source") else []),
    "_language": ("token", lambda r: [(None, r["language"])] if r.get("language") else []),
}

PARAMS = {
    "Observation": {
        "patient": ("reference", _patient_ref), "subject": ("reference", _subject),
        "code": ("token", lambda r: codings(r.get("code"))), "code:text": ("text", lambda r: cc_texts(r.get("code"))),
        "category": ("token", lambda r: codings(r.get("category") or [])),
        "category:text": ("text", lambda r: cc_texts(r.get("category") or [])),
        "combo-code": ("token", lambda r: codings(r.get("code")) + [c for comp in r.get("component") or [] for c in codings(comp.get("code"))]),
        "component-code": ("token", lambda r: [c for comp in r.get("component") or [] for c in codings(comp.get("code"))]),
        "date": ("date", lambda r: period_or_dt(r, "effective")),
        "status": ("token", _status),
        "value-quantity": ("quantity", lambda r: _q(r.get("valueQuantity"))),
        "combo-value-quantity": ("quantity", lambda r: _q(r.get("valueQuantity")) + [q for comp in r.get("component") or [] for q in _q(comp.get("valueQuantity"))]),
        "component-value-quantity": ("quantity", lambda r: [q for comp in r.get("component") or [] for q in _q(comp.get("valueQuantity"))]),
        "value-string": ("string", lambda r: [r["valueString"]] if isinstance(r.get("valueString"), str) else []),
        "value-concept": ("token", lambda r: codings(r.get("valueCodeableConcept") or [])),
        "combo-value-concept": ("token", lambda r: codings(r.get("valueCodeableConcept") or []) + [c for comp in r.get("component") or [] for c in codings(comp.get("valueCodeableConcept") or [])]),
        "component-value-concept": ("token", lambda r: [c for comp in r.get("component") or [] for c in codings(comp.get("valueCodeableConcept") or [])]),
        "value-date": ("date", lambda r: period_or_dt(r, "value")),
        "data-absent-reason": ("token", lambda r: codings(r.get("dataAbsentReason") or [])),
        "combo-data-absent-reason": ("token", lambda r: codings(r.get("dataAbsentReason") or []) + [c for comp in r.get("component") or [] for c in codings(comp.get("dataAbsentReason") or [])]),
        "component-data-absent-reason": ("token", lambda r: [c for comp in r.get("component") or [] for c in codings(comp.get("dataAbsentReason") or [])]),
        "method": ("token", lambda r: codings(r.get("method") or [])),
        "identifier": ("token", _ident),
        "encounter": ("reference", lambda r: ref_values(r.get("encounter"))),
        "performer": ("reference", lambda r: ref_values(r.get("performer") or [])),
        "based-on": ("reference", lambda r: ref_values(r.get("basedOn") or [])),
        "derived-from": ("reference", lambda r: ref_values(r.get("derivedFrom") or [])),
        "has-member": ("reference", lambda r: ref_values(r.get("hasMember") or [])),
        "part-of": ("reference", lambda r: ref_values(r.get("partOf") or [])),
        "focus": ("reference", lambda r: ref_values(r.get("focus") or [])),
        "specimen": ("reference", lambda r: ref_values(r.get("specimen"))),
        "device": ("reference", lambda r: ref_values(r.get("device"))),
    },
    "Condition": {
        "patient": ("reference", _patient_ref), "subject": ("reference", _subject),
        "code": ("token", lambda r: codings(r.get("code"))), "code:text": ("text", lambda r: cc_texts(r.get("code"))),
        "category": ("token", lambda r: codings(r.get("category") or [])),
        "clinical-status": ("token", lambda r: codings(r.get("clinicalStatus") or [])),
        "verification-status": ("token", lambda r: codings(r.get("verificationStatus") or [])),
        "severity": ("token", lambda r: codings(r.get("severity") or [])),
        "body-site": ("token", lambda r: codings(r.get("bodySite") or [])),
        "stage": ("token", lambda r: [c for s in r.get("stage") or [] for c in codings(s.get("summary") or [])]),
        "evidence": ("token", lambda r: [c for e in r.get("evidence") or [] for c in codings(e.get("code") or [])]),
        "onset-date": ("date", lambda r: period_or_dt(r, "onset")),
        "recorded-date": ("date", lambda r: _dates(r.get("recordedDate"))),
        "abatement-date": ("date", lambda r: period_or_dt(r, "abatement")),
        "onset-age": ("quantity", lambda r: _q(r.get("onsetAge"))),
        "abatement-age": ("quantity", lambda r: _q(r.get("abatementAge"))),
        "onset-info": ("string", lambda r: [r["onsetString"]] if isinstance(r.get("onsetString"), str) else []),
        "abatement-string": ("string", lambda r: [r["abatementString"]] if isinstance(r.get("abatementString"), str) else []),
        "identifier": ("token", _ident),
        "encounter": ("reference", lambda r: ref_values(r.get("encounter"))),
        "asserter": ("reference", lambda r: ref_values(r.get("asserter"))),
        "evidence-detail": ("reference", lambda r: [x for e in r.get("evidence") or [] for x in ref_values(e.get("detail") or [])]),
    },
    "MedicationRequest": {
        "patient": ("reference", _patient_ref), "subject": ("reference", _subject),
        "status": ("token", _status), "intent": ("token", lambda r: [(None, r["intent"])] if r.get("intent") else []),
        "priority": ("token", lambda r: [(None, r["priority"])] if r.get("priority") else []),
        "category": ("token", lambda r: codings(r.get("category") or [])),
        "code": ("token", lambda r: codings(r.get("medicationCodeableConcept") or [])),
        "code:text": ("text", lambda r: cc_texts(r.get("medicationCodeableConcept") or [])),
        "medication": ("reference", lambda r: ref_values(r.get("medicationReference"))),
        "authoredon": ("date", lambda r: _dates(r.get("authoredOn"))),
        "date": ("date", lambda r: _dates(*[e for di in r.get("dosageInstruction") or [] for e in (di.get("timing") or {}).get("event") or []])),
        "identifier": ("token", _ident),
        "requester": ("reference", lambda r: ref_values(r.get("requester"))),
        "encounter": ("reference", lambda r: ref_values(r.get("encounter"))),
        "intended-dispenser": ("reference", lambda r: ref_values((r.get("dispenseRequest") or {}).get("performer"))),
        "intended-performer": ("reference", lambda r: ref_values(r.get("performer"))),
        "intended-performertype": ("token", lambda r: codings(r.get("performerType") or [])),
    },
    "Procedure": {
        "patient": ("reference", _patient_ref), "subject": ("reference", _subject),
        "code": ("token", lambda r: codings(r.get("code"))), "code:text": ("text", lambda r: cc_texts(r.get("code"))),
        "date": ("date", lambda r: period_or_dt(r, "performed")),
        "status": ("token", _status),
        "category": ("token", lambda r: codings(r.get("category") or [])),
        "reason-code": ("token", lambda r: codings(r.get("reasonCode") or [])),
        "identifier": ("token", _ident),
        "based-on": ("reference", lambda r: ref_values(r.get("basedOn") or [])),
        "encounter": ("reference", lambda r: ref_values(r.get("encounter"))),
        "location": ("reference", lambda r: ref_values(r.get("location"))),
        "part-of": ("reference", lambda r: ref_values(r.get("partOf") or [])),
        "performer": ("reference", lambda r: [x for p in r.get("performer") or [] for x in ref_values(p.get("actor"))]),
        "reason-reference": ("reference", lambda r: ref_values(r.get("reasonReference") or [])),
        "instantiates-canonical": ("reference", lambda r: list(r.get("instantiatesCanonical") or [])),
        "instantiates-uri": ("uri", lambda r: list(r.get("instantiatesUri") or [])),
    },
    "Patient": {
        "identifier": ("token", _ident),
        "name": ("string", lambda r: _names(r, "name")), "given": ("string", lambda r: _names(r, "given")),
        "family": ("string", lambda r: _names(r, "family")), "phonetic": ("string", lambda r: _names(r, "name")),
        "birthdate": ("date", lambda r: _dates(r.get("birthDate"))),
        "death-date": ("date", lambda r: _dates(r.get("deceasedDateTime"))),
        "deceased": ("token", lambda r: [(None, "true" if (r.get("deceasedBoolean") or r.get("deceasedDateTime")) else "false")]),
        "gender": ("token", lambda r: [(None, r["gender"])] if r.get("gender") else []),
        "active": ("token", lambda r: [(None, "true" if r["active"] else "false")] if "active" in r else []),
        "address": ("string", lambda r: _addr(r, "address")), "address-city": ("string", lambda r: _addr(r, "city")),
        "address-state": ("string", lambda r: _addr(r, "state")), "address-postalcode": ("string", lambda r: _addr(r, "postalCode")),
        "address-country": ("string", lambda r: _addr(r, "country")),
        "address-use": ("token", lambda r: [(None, a["use"]) for a in r.get("address") or [] if a.get("use")]),
        "phone": ("token", lambda r: _telecom(r, "phone")), "email": ("token", lambda r: _telecom(r, "email")),
        "telecom": ("token", lambda r: _telecom(r)),
        "language": ("token", lambda r: [c for x in r.get("communication") or [] for c in codings(x.get("language") or [])]),
        "general-practitioner": ("reference", lambda r: ref_values(r.get("generalPractitioner") or [])),
        "organization": ("reference", lambda r: ref_values(r.get("managingOrganization"))),
        "link": ("reference", lambda r: [x for l in r.get("link") or [] for x in ref_values(l.get("other"))]),
    },
}
APPROXIMATE_KINDS = {"composite", "special"}
WARM = {"Observation": ["code", "category", "status", "date"], "Procedure": ["code", "date"], "Condition": ["code", "category"],
        "MedicationRequest": ["status", "category", "code", "date", "authoredon"], "Patient": ["identifier", "name", "given", "family", "birthdate", "gender"]}
SUBSETTED = {"system": "http://terminology.hl7.org/CodeSystem/v3-ObservationValue", "code": "SUBSETTED",
             "display": "Resource encoded in summary mode"}


def hapi_quote(s):
    return urllib.parse.quote(s, safe="*").replace("~", "%7E")


# ------------------------------------------------------------------------------------------------ the store

class Store:
    def __init__(self, snapshot: Path):
        t0 = time.time()
        self.snapshot = snapshot
        self.manifest = json.loads((snapshot / "manifest.json").read_text())
        self.declared = json.loads((snapshot / "search_params.json").read_text())   # type -> {param: kind}
        self.types_lower = {x.lower() for x in self.declared}       # HAPI resolves resource type names case-blind
        self.metadata_text = (snapshot / "metadata.json").read_text()
        self.res, self.text, self.by_id = {}, {}, {}
        for t in DATA_TYPES:
            rs, tx = [], []
            with gzip.open(snapshot / f"{t}.jsonl.gz", "rt") as f:
                for line in f:
                    r = loads(line)
                    rs.append(r)
                    tx.append(dumps_pretty(r, 2))          # as it sits in a search bundle's entry
            self.res[t], self.text[t] = rs, tx
            self.by_id[t] = {r["id"]: i for i, r in enumerate(rs)}
        # HAPI's resource PID order: numeric ids are the PIDs; Patient ids are client-assigned, their PID order was
        # read from HAPI (a :missing search returns every patient in PID order) - patient_pid_order.json
        porder = {p: k for k, p in enumerate(json.loads((snapshot / "patient_pid_order.json").read_text()))}
        self.pid = {t: [porder[r["id"]] if t == "Patient" else int(r["id"]) for r in self.res[t]] for t in DATA_TYPES}
        self.by_patient = {t: defaultdict(list) for t in DATA_TYPES}
        for t in DATA_TYPES:
            for i, r in enumerate(self.res[t]):
                pid = r["id"] if t == "Patient" else (_patient_ref(r)[0].split("/", 1)[1] if _patient_ref(r) else None)
                if pid is not None:
                    self.by_patient[t][pid].append(i)
            for lst in self.by_patient[t].values():
                lst.sort(key=lambda i: self.pid[t][i])
        self._cache = {}                        # (type, param) -> per-resource extracted values; token indexes
        self._lock = threading.Lock()
        for t, ps in WARM.items():
            for p_ in ps:
                self.values(t, p_)
                if (PARAMS[t].get(p_) or COMMON.get(p_))[0] == "token":
                    self.token_index(t, p_)
        self.pages = OrderedDict()              # search id -> (type, indices, params for links)
        self.stamp = "2026-10-04T00:00:00.000+00:00"
        log(f"mock FHIR: loaded {sum(len(v) for v in self.res.values())} resources in {time.time() - t0:.0f}s")

    def values(self, t, p):
        key = (t, p)
        v = self._cache.get(key)
        if v is None:
            kind, fn = (PARAMS[t].get(p) or COMMON[p])
            v = [fn(r) for r in self.res[t]]
            with self._lock:
                self._cache[key] = v
        return v

    # ---------------------------------------------------------------------------------------------- search
    def search(self, t, params):
        """params: list of (name, value) after URL decoding; returns (indices, control dict)."""
        ctl = {"count": None, "sort": [], "summary": None, "total": None, "elements": None, "include": [], "revinclude": []}
        declared = self.declared.get(t, {})
        raw = []
        for name, value in params:
            base = name.split(":", 1)[0]
            if base in FULLTEXT:
                raise Bad(400, "HAPI-1192", f"Fulltext search is not enabled on this service, can not process parameter: {base}")
            if base == "_count":
                try:
                    n = int(value)
                    if not (-(1 << 31) <= n < (1 << 31)):
                        raise ValueError
                except ValueError:
                    raise Bad(400, "HAPI-0375", f"Invalid _count value: {value}")
                ctl["count"] = n
                continue
            if base == "_sort":
                for k in value.split(","):
                    k = k.strip()
                    desc = k.startswith("-")
                    key = k[1:] if desc else k
                    if key not in declared and key not in ("_id", "_lastUpdated", "_score"):
                        raise Bad(400, "HAPI-1194", f'Unknown _sort parameter value "{key}" for resource type "{t}"')
                    ctl["sort"].append((key, desc))
                continue
            if base == "_summary":
                ctl["summary"] = value
                continue
            if base == "_total":
                ctl["total"] = value
                continue
            if base == "_elements":
                ctl["elements"] = [e.strip() for e in value.split(",") if e.strip()]
                continue
            if base in ("_include", "_revinclude"):
                if value != "*":
                    parts = value.split(":")
                    if len(parts) < 2 or parts[0] not in self.declared or parts[1] not in self.declared[parts[0]]:
                        raise Bad(400, "HAPI-2015", f'Invalid {base} parameter value: "{value}".')
                ctl["include" if base == "_include" else "revinclude"].append(value)
                continue
            if base.startswith("_") and base not in declared:
                continue                              # _since, _until, _lastN, _securityContext, unknown: ignored
            if "." in name:                           # chained parameter
                raw.append(("approx", name, value))
                continue
            if base not in declared:
                raise Bad(400, "HAPI-0524", f'Unknown search parameter "{base}" for resource type "{t}".')
            kind, mod = declared[base], name.partition(":")[2]
            if kind == "reference" and mod and mod not in REF_MODIFIERS and mod.lower() not in self.types_lower:
                raise Bad(400, "HAPI-2498", f'Unsupported search modifier(s): "[:{mod}]" for resource type "{t}".')
            raw.append((kind, name, value))
        filters = self.parse_filters(t, raw)        # every value is parsed (and refused) before anything is matched
        for kind, name, value, _ in filters:
            mod = name.partition(":")[2]
            if kind == "approx" or kind in APPROXIMATE_KINDS or mod in ("above", "below", "in", "not-in", "of-type", "iterate"):
                log(f"mock FHIR approximate (answered as no match): {t}?{name}={value}")
        if t not in DATA_TYPES:
            return [], ctl
        idx = self.candidates(t, filters)
        for f in filters:
            idx = [i for i in idx if self.match(t, f, i)]
        if filters:                                     # HAPI returns rows in the order of the driving index scan
            idx = self.default_order(t, list(idx), filters)
        if ctl["sort"]:
            idx = self.sort(t, list(idx), ctl["sort"])  # stable: ties keep the driving index order
        return list(idx), ctl

    def candidates(self, t, filters):
        """a superset of the matches from the indexes (patient references, token values), smallest first; every
        filter is still applied afterwards"""
        lists = []
        for kind, name, value, parsed in filters:
            base, _, mod = name.partition(":")
            if kind == "reference" and base in ("patient", "subject") and not mod and t != "Patient":
                # every subject in the data is a Patient reference: other types / absolute URLs match nothing
                lists.append({i for x in parsed if x is not None and x[0] in (None, "Patient")
                              for i in self.by_patient[t].get(x[1], ())})
            elif kind == "token" and not mod and (base in PARAMS[t] or base in COMMON):
                ix = self.token_index(t, base)
                out = set()
                for a in parsed:
                    if "|" in a:
                        sysm, code = a.split("|", 1)
                        key = ("s", sysm) if code == "" else ("sc", sysm or None, code)
                    else:
                        key = ("c", a)
                    out.update(ix.get(key, ()))
                lists.append(out)
        if not lists:
            return range(len(self.res[t]))
        lists.sort(key=len)
        best = lists[0]
        for other in lists[1:]:
            best = best & other
        return sorted(best)

    def token_index(self, t, p):
        key = ("tokix", t, p)
        ix = self._cache.get(key)
        if ix is None:
            ix = defaultdict(list)
            for i, vs in enumerate(self.values(t, p)):
                for sysm, code in vs:
                    for k in (("c", code), ("sc", sysm, code), ("s", sysm)):
                        lst = ix[k]
                        if not lst or lst[-1] != i:
                            lst.append(i)
            ix = dict(ix)
            with self._lock:
                self._cache[key] = ix
        return ix

    def parse_filters(self, t, raw):
        """[(kind, name, value)] -> [(kind, name, value, parsed)], raising what HAPI's parameter parsers raise: dates
        through DateRangeParam (one per parameter, every modifier), quantities through QuantityParam, references
        through ReferenceParam / IdDt."""
        dates = defaultdict(list)
        for kind, name, value in raw:
            if kind == "date":
                dates[name.split(":")[0]].append((name.partition(":")[2], value))
        ranges = {b: date_range_param(items) for b, items in dates.items()}
        out = []
        for kind, name, value in raw:
            base, _, mod = name.partition(":")
            if kind == "date":
                if mod == "missing":
                    out.append((kind, name, value, None))
                elif base in ranges:
                    if ranges[base]:
                        out.append((kind, base, value, ranges[base]))
                    del ranges[base]
                continue
            if value == "":
                if kind == "reference" and mod != "missing":
                    raise Bad(400, "HAPI-1101", "Parameter value missing in request")
                continue
            if (kind == "string" or (kind == "token" and mod == "text")) and mod != "missing":
                for a in split_or(value):
                    if len(a) > MAX_STRING:
                        raise Bad(400, "HAPI-1260", f"Parameter[{base}] has length ({len(a)}) that is longer than maximum allowed ({MAX_STRING}): {a}")
            if mod == "missing" or kind == "approx":
                parsed = None
            elif kind == "reference":
                parsed = [self.ref_target(t, base, mod, a) for a in split_or(value)]
            elif kind in ("quantity", "number"):
                parsed = [quantity_param(a) for a in split_or(value)]
            else:
                parsed = value.split(",")
            out.append((kind, name, value, parsed))
        return out

    def ref_target(self, t, base, mod, a):
        """one OR value of a reference parameter -> (type | None, id), or None when nothing can match (an absolute
        URL: HAPI searches those as external references, and every reference in the data is relative)."""
        type_mod = mod and mod not in REF_MODIFIERS
        burl, rtype, idp = iddt(a)
        if burl:
            return None
        if type_mod:
            if t == "Observation" and base == "patient":
                return None                             # measured: HAPI returns nothing for Observation?patient:<Type>=
            rtype = rtype or mod
        elif rtype and rtype.lower() not in self.types_lower:
            raise Bad(400, "HAPI-1250", f'Invalid/unsupported resource type: "{rtype}"')
        if not idp:
            raise Bad(400, "HAPI-1101", "Parameter value missing in request")
        return rtype, idp

    def default_order(self, t, idx, filters):
        """The order of HAPI's driving index scan (measured): token / string / uri / quantity params drive first
        (alphabetical by name); otherwise any :missing / :not filter makes the resource table drive (unfiltered order,
        i.e. the snapshot's); otherwise a date param whose bounds are all day-or-coarser or all with a time and none
        'ne'; otherwise reference / _id; otherwise any other date param. Sort keys: token -> (hash of the matched value, PID); string -> (matched
        normalized value, PID); date -> (calendar day, PID) for day bounds, (instant, PID) for time bounds;
        quantity / uri -> (value, PID); reference -> PID; resource table -> snapshot position."""
        def cls(f):
            kind, name = f[0], f[1]
            mod = name.partition(":")[2]
            if mod in ("missing", "not"):            # NOT EXISTS / EXISTS against the resource table
                return 1
            if kind in ("reference", "approx") or name.split(":")[0] == "_id":
                return 3
            if kind == "date":                       # a range scan needs one kind of bound and no 'ne'; otherwise
                bounds = f[3] or []                  # the date index drives only after a reference
                return 2 if bounds and all(p != "ne" for p, _ in bounds) and len({b[4] for _, b in bounds}) == 1 else 4
            return 0
        driver = min(filters, key=lambda f: (cls(f), f[1].split(":")[0]))
        kind, name, value, parsed = driver
        if cls(driver) == 1:
            return sorted(idx)
        pid = self.pid[t]
        base, _, mod = name.partition(":")
        if cls(driver) == 3 or mod == "exact" or kind == "approx":
            return sorted(idx, key=lambda i: pid[i])
        alts = value.split(",")
        if kind == "token" and mod != "text":
            hashes = []
            for a in alts:
                if "|" in a:
                    sysm, code = a.split("|", 1)
                    h = hapi_hash(t, base, sysm) if code == "" else hapi_hash(t, base, sysm, code)
                else:
                    h = hapi_hash(t, base, a)
                hashes.append((a, h))
            vals = self.values(t, base)

            def k(i):
                return (min((h for a, h in hashes if self.token_hit(a, vals[i])), default=0), pid[i])
            return sorted(idx, key=k)
        if kind == "string" or (kind == "token" and mod == "text"):
            vals = self.values(t, base + ":text" if kind == "token" else base)
            pre = [norm_str(a) for a in alts]

            def k(i):
                ms = [norm_str(v) for v in vals[i]]
                hit = [v for v in ms if any(v.startswith(x) or (mod == "contains" and x in v) for x in pre)]
                return (min(hit) if hit else "", pid[i])
            return sorted(idx, key=k)
        if kind == "date":                           # time bounds scan (instant, PID), day bounds (day, PID)
            vals, col = self.values(t, base), 0 if all(b[4] for _, b in parsed) else 2

            def k(i):
                return (min((x[col] for x in vals[i]), default=0), pid[i])
            return sorted(idx, key=k)
        if kind == "quantity":
            vals = self.values(t, base)
            return sorted(idx, key=lambda i: (min((Decimal(str(x[0])) for x in vals[i] if x[0] is not None), default=Decimal(0)), pid[i]))
        if kind == "uri":
            vals = self.values(t, base)
            return sorted(idx, key=lambda i: (min(vals[i], default=""), pid[i]))
        return sorted(idx, key=lambda i: pid[i])

    def match(self, t, f, i):
        kind, name, value, parsed = f
        base, _, mod = name.partition(":")
        if kind == "approx" or kind in APPROXIMATE_KINDS or (mod and mod in ("above", "below", "in", "not-in", "of-type", "iterate")):
            return False
        if mod == "missing":
            present = bool(self.values(t, base)[i]) if (base in PARAMS[t] or base in COMMON) else False
            return present != (value.lower() == "true")
        if base not in PARAMS[t] and base not in COMMON:
            return False                                # declared, element never present in the data
        vals = self.values(t, base)[i]
        if kind == "token":
            if mod == "text":
                tv = [norm_str(x) for x in self.values(t, base + ":text")[i]] if (base + ":text") in PARAMS[t] else []
                return any(any(x.startswith(norm_str(a)) for x in tv) for a in parsed)
            hit = any(self.token_hit(a, vals) for a in parsed)
            return (not hit) if mod == "not" else hit
        if kind == "string":
            if mod == "exact":
                return any(v == a for a in parsed for v in vals)
            if mod == "contains":
                return any(norm_str(a) in norm_str(v) for a in parsed for v in vals)
            return any(norm_str(v).startswith(norm_str(a)) for a in parsed for v in vals)
        if kind == "date":                              # every bound must hold
            return all(any(date_match(p, s, tv) for tv in vals) for p, s in parsed)
        if kind == "reference":
            for tgt in parsed:
                if tgt is None:
                    continue
                rtype, idp = tgt
                if rtype:
                    if f"{rtype}/{idp}" in vals:
                        return True
                elif any(v.rsplit("/", 1)[-1] == idp for v in vals):
                    return True
            return False
        if kind in ("quantity", "number"):
            for prefix, num, system, code in parsed:
                for x, s, c, u in vals:
                    if code is not None and (c != code or (system is not None and s != system)):
                        continue                        # HAPI ignores a system given without units
                    if num is None or num_match(prefix, num, x):
                        return True
            return False
        if kind == "uri":
            return any(v == a for a in parsed for v in vals)
        return False

    @staticmethod
    def token_hit(a, vals):
        if "|" in a:
            sysm, code = a.split("|", 1)
            for s, c in vals:
                if sysm == "" and s is None and c == code:
                    return True
                if sysm and s == sysm and (code == "" or c == code):
                    return True
            return False
        return any(c == a for _, c in vals)

    def sort(self, t, idx, keys):
        def keyf(i):
            out = []
            for key, desc in keys:
                if key == "_id":
                    v = self.res[t][i]["id"]
                elif key in ("_lastUpdated",) or (key in PARAMS[t] and PARAMS[t][key][0] == "date") or key in COMMON and COMMON[key][0] == "date":
                    vs = self.values(t, key)[i]
                    v = min(x[0] for x in vs) if vs else None
                elif key in PARAMS[t] and PARAMS[t][key][0] == "token":
                    vs = self.values(t, key)[i]
                    v = min((c or "") for _, c in vs) if vs else None
                elif key in PARAMS[t] and PARAMS[t][key][0] == "string":
                    vs = self.values(t, key)[i]
                    v = min(norm_str(x) for x in vs) if vs else None
                elif key in PARAMS[t] and PARAMS[t][key][0] == "quantity":
                    vs = self.values(t, key)[i]
                    v = min(Decimal(str(x[0])) for x in vs if x[0] is not None) if vs else None
                else:
                    v = None
                out.append((v is None, v, desc))
            return out
        decorated = [(keyf(i), pos, i) for pos, i in enumerate(idx)]

        def cmp_key(item):
            ks, pos, i = item
            res = []
            for missing, v, desc in ks:
                if missing:
                    res.append((1, 0))               # resources without a value sort last either way
                else:
                    res.append((0, _Rev(v) if desc else v))
            res.append(pos)                          # ties keep the driving index order
            return res
        decorated.sort(key=cmp_key)
        return [i for _, _, i in decorated]

    # --------------------------------------------------------------------------------------------- bundles
    def includes(self, t, idx, ctl):
        out = []
        seen = set()
        for inc in ctl["include"]:
            parts = inc.split(":")
            if parts[0] != t:
                continue
            if parts[1] in ("patient", "subject") and t in DATA_TYPES and t != "Patient":
                for i in idx:
                    for ref in _patient_ref(self.res[t][i]):
                        pid = ref.split("/", 1)[1]
                        j = self.by_id["Patient"].get(pid)
                        if j is not None and ("Patient", j) not in seen:
                            seen.add(("Patient", j))
                            out.append(("Patient", j))
        for inc in ctl["revinclude"]:
            parts = inc.split(":")
            src, prm = parts[0], parts[1]
            if t == "Patient" and src in DATA_TYPES and src != "Patient" and prm in ("patient", "subject"):
                for i in idx:
                    for j in self.by_patient[src].get(self.res["Patient"][i]["id"], []):
                        if (src, j) not in seen:
                            seen.add((src, j))
                            out.append((src, j))
        return out

    def resource_obj(self, t, i, ctl):
        r = self.res[t][i]
        if not ctl.get("elements"):
            return r
        keep = {"resourceType", "id", "meta"} | set(ctl["elements"])
        sub = {k: v for k, v in r.items() if k in keep}
        meta = dict(sub.get("meta") or {})
        meta["tag"] = [SUBSETTED]
        sub["meta"] = meta
        return sub

    def resource_text(self, t, i, ctl):
        """pretty JSON of a resource as it sits in a search bundle's entry (pre-rendered unless subsetted)"""
        return dumps_pretty(self.resource_obj(t, i, ctl), 2) if ctl.get("elements") else self.text[t][i]

    def bundle(self, base, t, idx, ctl, self_url, sid, offset, count, first_page, qd):
        fmt, pretty = response_format(qd)
        n = len(idx)
        page = idx[offset: offset + count] if count > 0 else []
        has_next = count > 0 and offset + count < n
        if ctl.get("total") == "accurate" or count == 0:
            total = n
        elif ctl.get("total") == "none":
            total = None
        elif first_page and (n < PREFETCH or offset + count > n) and count > 0:
            total = n          # complete on the first page (exactly a full page: HAPI cannot know yet)
        elif not first_page:
            total = n          # HAPI: once the background search has finished (it has by the time an agent reads
        else:                  # page 1 and asks for page 2; an immediate scripted follow-up may still see none)
            total = None
        if count < 0:
            page, has_next, total = [], False, None
        tail = link_tail(qd, pretty)
        links = [("self", self_url)] if count != 0 else []      # HAPI: a count-only bundle has no links
        if has_next:
            links.append(("next", f"{base}?_getpages={sid}&_getpagesoffset={offset + count}&_count={count}{tail}"))
        if not first_page and offset > 0:
            links.append(("previous", f"{base}?_getpages={sid}&_getpagesoffset={max(0, offset - count)}&_count={count}{tail}"))
        entries = [(t, i, "match") for i in page] + [(it, j, "include") for it, j in self.includes(t, page, ctl)]
        if fmt != "json" or not pretty:
            d = {"resourceType": "Bundle", "id": sid, "meta": {"lastUpdated": self.stamp}, "type": "searchset"}
            if count == 0:
                d["meta"]["tag"] = [SUBSETTED]
            if total is not None:
                d["total"] = total
            if links:
                d["link"] = [{"relation": rel, "url": url} for rel, url in links]
            if entries:
                d["entry"] = [{"fullUrl": f"{base}/{et}/{self.res[et][i]['id']}", "resource": self.resource_obj(et, i, ctl),
                               "search": {"mode": mode}} for et, i, mode in entries]
            return render(d, fmt, pretty)
        head = ["{", '\n  "resourceType": "Bundle"', f',\n  "id": "{sid}"', f',\n  "meta": {{\n    "lastUpdated": "{self.stamp}"'
                + (',\n    "tag": [ {\n      "system": "http://terminology.hl7.org/CodeSystem/v3-ObservationValue",\n      '
                   '"code": "SUBSETTED",\n      "display": "Resource encoded in summary mode"\n    } ]' if count == 0 else "")
                + "\n  }", ',\n  "type": "searchset"']
        if total is not None:
            head.append(f',\n  "total": {total}')
        if links:
            head.append(',\n  "link": [ ' + ", ".join(
                "{\n    \"relation\": \"%s\",\n    \"url\": %s\n  }" % (rel, json.dumps(url, ensure_ascii=False)) for rel, url in links) + " ]")
        if entries:
            head.append(',\n  "entry": [ ' + ", ".join(
                "{\n    \"fullUrl\": " + json.dumps(f"{base}/{et}/{self.res[et][i]['id']}") + ",\n    \"resource\": "
                + self.resource_text(et, i, ctl) + ",\n    \"search\": {\n      \"mode\": \"%s\"\n    }\n  }" % mode
                for et, i, mode in entries) + " ]")
        head.append("\n}")
        return "".join(head)

    def respond(self, path, query, host):
        """-> (status, body text, content type); errors are OperationOutcomes in the requested encoding"""
        qd = defaultdict(list)
        for k, v in urllib.parse.parse_qsl(query, keep_blank_values=True):
            qd[k].append(v)
        fmt, pretty = response_format(qd)
        try:
            status, body = self.handle(path, query, host, qd)
        except Bad as e:
            status, body = e.status, render(outcome(e.code, e.msg), fmt, pretty)
        return status, body, CONTENT_TYPES[fmt]

    def handle(self, path, query, host, qd):
        base = f"http://{host}/fhir"
        params = urllib.parse.parse_qsl(query, keep_blank_values=True)
        fmt, pretty = response_format(qd)
        p = path.split("?", 1)[0]
        if not p.startswith("/fhir"):
            raise Bad(404, "HAPI-0000", "Not found")
        rest = p[len("/fhir"):].strip("/")
        if rest == "metadata":
            return 200, self.metadata_text if fmt == "json" and pretty else render(loads(self.metadata_text), fmt, pretty)
        if rest == "" and "_getpages" in qd:
            sid = qd["_getpages"][0]
            with self._lock:
                got = self.pages.get(sid)
            if got is None:
                raise Bad(410, "HAPI-1124", f"Search ID \"{sid}\" does not exist and may have expired")
            t, idx, ctl = got
            offset = int(qd.get("_getpagesoffset", ["0"])[0])
            count = int(qd.get("_count", [str(DEFAULT_COUNT)])[0])
            return 200, self.bundle(base, t, idx, ctl, f"{base}?{query}", sid, offset, count, False, qd)
        parts = [x for x in rest.split("/") if x]
        if not parts:
            raise Bad(400, "HAPI-0000", "This is the base URL of FHIR server. Unable to handle this request, as it does not contain a resource type or operation name.")
        t = parts[0]
        if t not in self.declared:
            raise Bad(404, "HAPI-0302", f"Unknown resource type '{t}' - Server knows how to handle: [...]")
        if len(parts) >= 2:                              # read, vread (Type/id/_history/n), instance history
            rid = parts[1]
            i = self.by_id[t].get(rid) if t in DATA_TYPES else None
            if i is None or len(parts) > 4 or (len(parts) > 2 and parts[2] != "_history"):
                raise Bad(404, "HAPI-1996", f"Resource {t}/{rid} is not known")
            r = self.res[t][i]
            ver = (r.get("meta") or {}).get("versionId", "1")
            if len(parts) == 2:
                return 200, render(r, fmt, pretty)
            if len(parts) == 4:
                if parts[3] != ver:
                    raise Bad(404, "HAPI-0979", f'Version "{parts[3]}" is not valid for resource {t}/{rid}')
                return 200, render(r, fmt, pretty)
            hist = {"resourceType": "Bundle", "id": str(uuid.uuid5(NS, path + "?" + query)), "meta": {"lastUpdated": self.stamp},
                    "type": "history", "total": 1, "link": [{"relation": "self", "url": f"{base}/{t}/{rid}/_history" + (f"?{query}" if query else "")}],
                    "entry": [{"fullUrl": f"{base}/{t}/{rid}", "resource": r, "request": {"method": "POST", "url": f"{t}/{rid}/_history/{ver}"},
                               "response": {"status": "201 Created", "etag": f'W/"{ver}"'}}]}
            return 200, render(hist, fmt, pretty)
        idx, ctl = self.search(t, params)
        count = ctl["count"] if ctl["count"] is not None else DEFAULT_COUNT
        if ctl["summary"] == "count":
            count = 0
        sid = str(uuid.uuid5(NS, t + "?" + "&".join(f"{k}={v}" for k, v in params)))
        order = sorted(range(len(params)), key=lambda j: params[j][0])
        self_url = f"{base}/{t}?" + "&".join(f"{hapi_quote(params[j][0])}={hapi_quote(params[j][1])}" for j in order) if params else f"{base}/{t}"
        if 0 < count < len(idx):
            with self._lock:
                self.pages[sid] = (t, idx, ctl)
                while len(self.pages) > 20000:
                    self.pages.popitem(last=False)
        return 200, self.bundle(base, t, idx, ctl, self_url, sid, 0, count, True, qd)


class _Rev:
    __slots__ = ("v",)

    def __init__(self, v):
        self.v = v

    def __lt__(self, o):
        return self.v > o.v

    def __eq__(self, o):
        return self.v == o.v


XML_FORMATS = {"xml", "text/xml", "application/xml", "application/fhir+xml", "application/xml+fhir"}
JSON_FORMATS = {"json", "application/json", "application/fhir+json", "application/json+fhir"}
CONTENT_TYPES = {"json": "application/fhir+json;charset=UTF-8", "xml": "application/fhir+xml;charset=UTF-8"}


def response_format(qd):
    """RestfulServerUtils: the first _format value HAPI recognises picks the encoding (JSON by default); the first
    _pretty value must be exactly "true" to pretty-print, no _pretty = pretty (the server's default)."""
    fmt = "json"
    for f in qd.get("_format", []):
        f = f.strip().replace(" ", "+")
        if f in JSON_FORMATS or f in XML_FORMATS:
            fmt = "json" if f in JSON_FORMATS else "xml"
            break
    return fmt, (qd["_pretty"][0] == "true" if qd.get("_pretty") else True)


def link_tail(qd, pretty):
    """what HAPI copies into next / previous links (RestfulServerUtils.createPagingLink)"""
    tail = ""
    if qd.get("_format"):
        tail += "&_format=" + escape(qd["_format"][0].replace(" ", "+"))
    if pretty:
        tail += "&_pretty=true"
    for v in qd.get("_include", []):
        if v.strip():
            tail += "&_include=" + escape(v)
    tail += "&_bundletype=searchset"
    for v in qd.get("_elements", []):
        if v.strip():
            tail += "&_elements=" + v
    return tail


def render(obj, fmt, pretty):
    if fmt == "xml":
        return dumps_xml(obj, 0, pretty)
    return dumps_pretty(obj, 0) if pretty else dumps_compact(obj)


def outcome(code, msg):
    return {"resourceType": "OperationOutcome", "issue": [{"severity": "error", "code": "processing", "diagnostics": f"{code}: {msg}"}]}


def make_handler(store):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True
        server_version = "Jetty(mock)"
        sys_version = ""

        def log_message(self, *a):
            pass

        def _send(self, status, body, ctype=CONTENT_TYPE):
            data = body.encode("utf-8")
            head = f"HTTP/1.1 {status} \r\nContent-Type: {ctype}\r\nContent-Length: {len(data)}\r\n\r\n"
            self.wfile.write(head.encode("latin-1") + data)     # one write: no Nagle / delayed-ACK stall

        def do_GET(self):
            if len(self.raw_requestline) + sum(len(k) + len(v) + 4 for k, v in self.headers.items()) + 2 > MAX_HEADER:
                self._send(400, "<h1>Bad Message 400</h1><pre>reason: Header too large</pre>", "text/html;charset=utf-8")
                return
            path, _, query = self.path.partition("?")
            try:
                status, body, ctype = store.respond(path, query, self.headers.get("Host") or "localhost")
            except Exception as e:  # noqa: BLE001
                log(f"mock FHIR error on {self.path}: {type(e).__name__}: {e}")
                status, body, ctype = 500, dumps_pretty(outcome("HAPI-0000", f"{type(e).__name__}: {e}")), CONTENT_TYPE
            self._send(status, body, ctype)

        def do_POST(self):
            self._send(405, dumps_pretty(outcome("HAPI-0000", "The mock FHIR server is read-only")))
    return H


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 1024              # listen backlog (must be set before the socket starts listening)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    store = Store(Path(a.snapshot))
    srv = Server((a.host, a.port), make_handler(store))
    log(f"mock FHIR serving on http://{a.host}:{a.port}/fhir/")
    srv.serve_forever()


if __name__ == "__main__":
    main()
