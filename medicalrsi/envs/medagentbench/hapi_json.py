"""JSON (and XML) exactly as HAPI FHIR 7.6 writes it (for the offline FHIR server).

The agent reads the raw response text, so the mock must reproduce HAPI's pretty printing (Jackson
DefaultPrettyPrinter with HAPI's separators): objects open with "{" + newline and indent 2 spaces per level, keys
written as `"key": value`; arrays are inline - "[ " + values joined by ", " + " ]" - and do not add an indent level,
so an object inside an array is indented one level deeper than the key that holds the array; empty containers are
"{ }" / "[ ]"; non-ASCII characters are written raw. Numbers keep their original text: decimals are parsed into
Decimal (str(Decimal("4.50")) == "4.50") so a resource written back is byte-identical to what HAPI sent.
"""
from __future__ import annotations

import json
from decimal import Decimal


def loads(text):
    return json.loads(text, parse_float=Decimal)


def _scalar(v):
    if v is True:
        return "true"
    if v is False:
        return "false"
    if v is None:
        return "null"
    if isinstance(v, Decimal):
        return str(v)
    if isinstance(v, (int, float)):
        return json.dumps(v)
    return json.dumps(v, ensure_ascii=False)


def dumps_pretty(v, nest=0):
    out = []

    def w(x, n):
        if isinstance(x, dict):
            if not x:
                out.append("{ }")
                return
            out.append("{")
            first = True
            for k, y in x.items():
                out.append(("\n" if first else ",\n") + "  " * (n + 1) + json.dumps(k, ensure_ascii=False) + ": ")
                first = False
                w(y, n + 1)
            out.append("\n" + "  " * n + "}")
        elif isinstance(x, list):
            if not x:
                out.append("[ ]")
                return
            out.append("[ ")
            for i, y in enumerate(x):
                if i:
                    out.append(", ")
                w(y, n)
            out.append(" ]")
        else:
            out.append(_scalar(x))
    w(v, nest)
    return "".join(out)


def dumps_compact(v):
    if isinstance(v, dict):
        return "{" + ",".join(json.dumps(k, ensure_ascii=False) + ":" + dumps_compact(y) for k, y in v.items()) + "}"
    if isinstance(v, list):
        return "[" + ",".join(dumps_compact(y) for y in v) + "]"
    return _scalar(v)


def _xattr(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace('"', "&quot;")


def _xtext(v):
    if v is True:
        return "true"
    if v is False:
        return "false"
    return str(v)


def dumps_xml(resource, level=0, pretty=True):
    """FHIR XML as HAPI writes it (for _format=xml): one element per line, 3 spaces per level, no XML declaration;
    primitives as <name value="..."/>; Element.id and Extension.url as attributes; a resource held by an element
    (Bundle.entry.resource, contained) is wrapped in its own element carrying the FHIR namespace; attribute values
    escape only & < " (as HAPI's StAX writer); elements in the JSON key order (both encoders walk the same model
    order). Primitive extensions ("_x") and narrative XHTML do not occur in the snapshot and are not handled."""
    out = []
    ind = (lambda n: "   " * n) if pretty else (lambda n: "")

    def res(r, n):
        t = r["resourceType"]
        out.append(f'{ind(n)}<{t} xmlns="http://hl7.org/fhir">')
        body(r, n + 1, True, False)
        out.append(f"{ind(n)}</{t}>")

    def body(obj, n, is_resource, is_ext):
        for k, v in obj.items():
            if k == "resourceType" or (k == "id" and not is_resource) or (k == "url" and is_ext):
                continue
            for x in v if isinstance(v, list) else [v]:
                if isinstance(x, dict) and "resourceType" in x:
                    out.append(f"{ind(n)}<{k}>")
                    res(x, n + 1)
                    out.append(f"{ind(n)}</{k}>")
                elif isinstance(x, dict):
                    ext = k in ("extension", "modifierExtension")
                    attrs = f' id="{_xattr(_xtext(x["id"]))}"' if "id" in x else ""
                    if ext and "url" in x:
                        attrs += f' url="{_xattr(x["url"])}"'
                    if any(kk != "id" and not (ext and kk == "url") for kk in x):
                        out.append(f"{ind(n)}<{k}{attrs}>")
                        body(x, n + 1, False, ext)
                        out.append(f"{ind(n)}</{k}>")
                    else:
                        out.append(f"{ind(n)}<{k}{attrs}/>")
                else:
                    out.append(f'{ind(n)}<{k} value="{_xattr(_xtext(x))}"/>')
    res(resource, level)
    return ("\n" if pretty else "").join(out)
