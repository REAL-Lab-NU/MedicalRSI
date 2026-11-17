"""The MedAgentGym (EHR-SeqSQL) environment's database side: the agent's SQL tool on a read-only SQLite copy of
MIMIC-III, tools as SQL views, action parsing and grading. Queries go through the post-processing of the official EHRSQL
scorer (the current time fixed at 2105-12-31 23:59:00); answers are compared as sets of normalised values."""
from __future__ import annotations

import ast
import json
import re
import sqlite3
import time
from pathlib import Path

NOW = "2105-12-31 23:59:00"
DB_NAME = "MIMIC-III"
MAX_ROWS_SHOWN, MAX_ROWS, MAX_CELL, SQL_SECONDS = 20, 1000, 100, 10
CODE_CHARS = 2000                    # a tool's views may be this long without counting toward the harness length
CODE = re.compile(r"```sql[ \t]*\n(.*?)\n?```", re.S | re.I)
VIEW = re.compile(r"(?is)^\s*create\s+temp(?:orary)?\s+view\s+(?:if\s+not\s+exists\s+)?([a-z_][a-z0-9_]*)\s+as\s+(?:select|with)\b")
CODE_NOTE = ("A tool's views are created before every query (only their names and columns are shown here): select "
             "from them by name in your one SELECT; do not write their joins again.\n")
_VITALS = {"temperature": (35.5, 38.1), "sao2": (95.0, 100.0), "heart rate": (60.0, 100.0), "respiration": (12.0, 18.0),
           "systolic bp": (90.0, 120.0), "diastolic bp": (60.0, 90.0), "mean bp": (60.0, 110.0)}


def postprocess(query: str) -> str:
    """post_process_sql of the official EHRSQL (2022, MIMIC-III) scorer: lower case, current_time and 'now' -> the
    fixed current time, the vital-sign range placeholders -> their normal ranges."""
    query = query.lower()
    if "current_time" in query:
        query = query.replace("current_time", f"'{NOW}'")
    if re.search('[ \n]+([a-zA-Z0-9_]+_lower)', query) and re.search('[ \n]+([a-zA-Z0-9_]+_upper)', query):
        lo = re.findall('[ \n]+([a-zA-Z0-9_]+_lower)', query)[0]
        hi = re.findall('[ \n]+([a-zA-Z0-9_]+_upper)', query)[0]
        names = list(set(re.findall('([a-zA-Z0-9_]+)_lower', lo) + re.findall('([a-zA-Z0-9_]+)_upper', hi)))
        if len(names) == 1 and names[0].replace('_', ' ') in _VITALS:
            r = _VITALS[names[0].replace('_', ' ')]
            query = query.replace(lo, f"{r[0]}").replace(hi, f"{r[1]}")
    query = query.replace("''", "'").replace('< =', '<=')
    query = query.replace("%y", "%Y").replace('%j', '%J')
    return query.replace("'now'", f"'{NOW}'")


def tool_code(text: str):
    """The first ```sql block of a harness file (its views), or None."""
    m = CODE.search(text or "")
    return (m.group(1).strip() or None) if m else None


def statements(code: str) -> list:
    """The statements of a block, split at the semicolons that end a complete statement."""
    out, cur = [], ""
    for part in (code or "").split(";"):
        cur += part + ";"
        if sqlite3.complete_statement(cur):
            if cur.strip(" ;\n\t"):
                out.append(cur.strip().rstrip(";").strip())
            cur = ""
    if cur.strip(" ;\n\t"):
        out.append(cur.strip().rstrip(";").strip())
    return out


def prelude(blocks: list) -> list:
    """[(harness name, statement)] of the views in the harness blocks [(harness name, code)]."""
    return [(name, st) for name, code in blocks for st in statements(code)]


class Database:
    def __init__(self, path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"MIMIC-III database not found: {self.path} (env.database in the config)")

    def connect(self, seconds: float):
        con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        con.text_factory = lambda b: b.decode(errors="ignore")
        t0 = time.time()
        con.set_progress_handler(lambda: int(time.time() - t0 > seconds), 10000)
        return con

    def schema(self) -> dict:
        con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        out = {}
        for (t,) in con.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"):
            out[t] = [f"{c[1]} {c[2]}".strip() for c in con.execute(f"PRAGMA table_info('{t}')")]
        con.close()
        return out

    def run_gold(self, sql: str, limit=None):
        """(rows, ok) of a gold query after the scorer's post-processing."""
        con = self.connect(60)
        try:
            cur = con.execute(postprocess(sql.strip().rstrip(";")))
            return (cur.fetchmany(limit) if limit else cur.fetchall()), True
        except Exception:  # noqa: BLE001
            return None, False
        finally:
            con.close()

    @staticmethod
    def create_views(con, views) -> list:
        """Creates the views on the connection; the notes of those refused or failed (the query still runs)."""
        notes = []
        for name, stmt in views:
            q = postprocess(stmt)
            m = VIEW.match(q)
            if not m:
                notes.append(f"note: a statement of skill {name} is not a CREATE TEMP VIEW and was skipped")
                continue
            try:
                con.execute(q)
            except Exception as e:  # noqa: BLE001
                notes.append(f"note: view {m.group(1)} of skill {name} failed: {str(e)[:150]}")
        return notes

    def tool_view(self, content: str) -> str:
        """A harness's content with each ```sql block reduced to its views' names and columns (the agent queries the
        views; their definitions stay hidden). Nothing is queried."""
        def view(m):
            con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
            lines = []
            try:
                for st in statements(m.group(1)):
                    q = postprocess(st)
                    mm = VIEW.match(q)
                    if not mm:
                        continue
                    try:
                        con.execute(q)
                        cols = [r[1] for r in con.execute(f"PRAGMA table_info('{mm.group(1)}')")]
                        lines.append(f"VIEW {mm.group(1)}({', '.join(cols)})")
                    except Exception:  # noqa: BLE001
                        lines.append(f"VIEW {mm.group(1)} (its definition does not work)")
            finally:
                con.close()
            return "```sql\n" + "\n".join(lines) + "\n```" if lines else m.group(0)
        return CODE.sub(view, content)

    def run_sql(self, sql: str, views=()):
        """(reply text, ok) of the agent's query; views [(harness name, CREATE TEMP VIEW statement)] are created first."""
        q = postprocess(sql.strip().rstrip(";"))
        if not re.match(r"(?is)^\s*(with|select)\b", q):
            return "SQL error: only a single SELECT (or WITH ... SELECT) statement is allowed.", False
        con = self.connect(SQL_SECONDS)
        notes = self.create_views(con, views) if views else []
        tail = ("\n" + "\n".join(notes)) if notes else ""
        try:
            rows = con.execute(q).fetchmany(MAX_ROWS + 1)
        except Exception as e:  # noqa: BLE001
            msg = "query interrupted: it ran longer than %d s" % SQL_SECONDS if "interrupted" in str(e) else str(e)
            return f"SQL error: {msg}" + tail, False
        finally:
            con.close()
        n = len(rows)
        shown = [[(c[:MAX_CELL] + "...") if isinstance(c, str) and len(c) > MAX_CELL else c for c in r]
                 for r in rows[:MAX_ROWS_SHOWN]]
        total = f"more than {MAX_ROWS}" if n > MAX_ROWS else str(n)
        more = f" (first {MAX_ROWS_SHOWN} shown)" if n > MAX_ROWS_SHOWN else ""
        return f"Result: {total} row(s){more}: {json.dumps(shown, default=str)}" + tail, True


def parse_action(text: str):
    """('sql', query) | ('finish', the text inside FINISH(...)) | ('invalid', text)."""
    t = (text or "").strip()
    t = re.sub(r"^```(?:sql)?\s*\n?(.*?)\n?```$", r"\1", t, flags=re.S).strip()
    if t.startswith("FINISH(") and t.endswith(")"):
        return "finish", t[len("FINISH("):-1]
    m = re.match(r"(?s)^SQL\s+(.+)$", t)
    if m:
        q = re.sub(r"^```(?:sql)?\s*\n?(.*?)\n?```$", r"\1", m.group(1).strip(), flags=re.S).strip()
        if re.search(r"(?m)^\s*FINISH\(", q):
            return "invalid", t                    # two actions in one message
        return "sql", q
    return "invalid", t


def norm(v) -> str:
    if isinstance(v, bool):
        return "1" if v else "0"
    try:
        return str(round(float(v), 3))
    except (TypeError, ValueError):
        s = str(v).strip().lower()
        s = re.sub(r"\s+", " ", s)                                   # runs of whitespace
        s = re.sub(r"^(\d{4}-\d{2}-\d{2}) 00:00:00$", r"\1", s)      # a midnight time after a date
        return s


def answer_values(raw):
    try:
        v = json.loads(raw)
    except (ValueError, TypeError):
        try:
            v = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(v, (list, tuple)):
        v = [v]
    flat = []
    for x in v:
        flat += list(x) if isinstance(x, (list, tuple)) else [x]
    return flat


def grade(task: dict, finish_raw, sql_ok: bool):
    """(solved, process): solved when the answer set equals the gold set; the process score is half the share of
    passed checks (a query ran, FINISH was called, the answer equals or overlaps the gold set)."""
    gold = {norm(c) for r in task["answer"] for c in r}
    pred_list = answer_values(finish_raw) if finish_raw is not None else None
    pred = {norm(c) for c in pred_list} if pred_list is not None else None
    if pred is not None and len(gold) == 1 and gold <= {"0.0", "1.0"}:      # yes / no questions
        yes, no = {"true", "yes", "1.0"}, {"false", "no", "0.0"}
        pred = {"1.0" if p in yes else "0.0" if p in no else p for p in pred}
    exact = pred is not None and pred == gold
    overlap = pred is not None and (bool(pred & gold) or (not pred and not gold))
    shown = json.dumps(task["answer"])[:120]
    groups = {"query": [["a query ran without error", bool(sql_ok)]],
              "action": [["called FINISH", finish_raw is not None]],
              "answer": [[f"answer = gold (expected {shown})", exact], ["answer overlaps the gold set", overlap]]}
    checks = [ok for cs in groups.values() for _, ok in cs]
    frac = sum(checks) / len(checks)
    return exact, {"score": 1.0 if exact else round(0.5 * frac, 4), "full": exact, "frac": round(frac, 4), "groups": groups}
