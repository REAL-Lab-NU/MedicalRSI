"""MedAgentGym, EHR-SeqSQL subset: clinical data questions on MIMIC-III. The Solver answers by querying a read-only SQLite
copy of the database (`SQL <query>` or `FINISH([...])`, at most 8 turns). A workflow is one SQL structure: 26 training
workflows with 20 training and 10 test tasks, and 6 unseen workflows with 10 test tasks (prepare.py builds the split).
The harness is a skill or a tool, i.e. SQL views with their documentation; the Solver sees each view's name and columns.
MIMIC-III is credentialed data (PhysioNet)."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests

from medicalrsi.envs.base import Env, render_harness, with_harness
from medicalrsi.envs.medagentgym import sql
from medicalrsi.utils import file_sha

BENCH = "EHRSQL"                       # the workflows' name prefix (EHRSQL/task1 ... EHRSQL/task44)
MAX_TURNS = 8
KINDS = ["ended by an invalid action", "used all 8 turns without calling FINISH", "ran out of context",
         "a query failed with an SQL error", "answered without querying the database"]


def parse_skill_file(text: str) -> dict:
    """{name, description, content} of a harness file (front matter read line by line)."""
    name, desc, content = "skill", "", text
    m = re.match(r"(?s)^---\s*\n(.*?)\n---\s*\n?(.*)$", text.strip())
    if m:
        content = m.group(2).strip()
        for line in m.group(1).splitlines():
            if line.startswith("name:"):
                name = line.split(":", 1)[1].strip().strip('"')
            elif line.startswith("description:"):
                desc = line.split(":", 1)[1].strip().strip('"')
    return {"name": name, "description": desc, "content": content}


def _sql_errors(r: dict) -> int:
    return sum(1 for m in r["history"][1:] if m["role"] != "agent" and (m["content"] or "").startswith("SQL error"))


def _queries(r: dict) -> int:
    return sum(1 for m in r["history"][1:] if m["role"] == "agent" and (m["content"] or "").lstrip().startswith("SQL"))


class MedAgentGym(Env):
    name = "medagentgym"
    benches = (BENCH,)
    improver_template = "improver_medagentgym.txt"
    has_tools = True
    code_block = sql.CODE
    code_chars = sql.CODE_CHARS
    write_tokens = 2048                # documentation and up to 2,000 characters of views
    min_scored_tasks = 10
    patient_id = re.compile(r"(?i)\b(?:patient|subject|hadm|admission)(?:_id)?\s*(?:id\s*)?[=#:]?\s*'?\d{2,6}\b")
    no_trace_problem = "no problem visible in the trace (the query logic or the reported answer was wrong)"

    def __init__(self, cfg):
        super().__init__(cfg)
        self.db = sql.Database(cfg.env.database)
        self.tasks = json.loads((self.data_dir / "tasks.json").read_text())
        self.types = json.loads((self.data_dir / "types.json").read_text())
        self._schema = None

    def load_split(self) -> dict:
        return json.loads((self.data_dir / "split.json").read_text())

    def heldout(self) -> set:
        return {(BENCH, t) for t, v in self.types["types"].items() if v["heldout"]}

    def train_order(self) -> list:
        return [tuple(x.split("/")) for x in self.types["train_order"]]

    # ------------------------------------------------------------ the Solver

    def prompt(self, task: dict) -> str:
        """The Solver's first message (the harness goes in behavioral_skills)."""
        if self._schema is None:
            self._schema = self.db.schema()
        context = (f"The EHR is the {sql.DB_NAME} database (SQLite). The current time is {sql.NOW}: use it for 'now', "
                   f"'today', 'this year' and other relative dates.")
        if task.get("seed_question"):
            context = (f"This question follows up on an earlier question in the same conversation: "
                       f"\"{task['seed_question'].strip()}\" The earlier question tells which patient, admission or item the "
                       f"follow-up is about. Answer only the follow-up question. " + context)
        return json.dumps({
            "phase": "task_execution",
            "task": {"description": task["question"], "context": context},
            "behavioral_skills": "",
            "api": {"database": f"{sql.DB_NAME}, SQLite, read-only",
                    "tables": {t: ", ".join(cols) for t, cols in self._schema.items()},
                    "functions": [{"name": "SQL <query>",
                                   "description": f"Runs one SQLite SELECT statement and returns at most {sql.MAX_ROWS_SHOWN} "
                                                  f"rows and the total row count. Write the whole query after 'SQL '."}]},
            "response_format": {"type": "api_action",
                                "options": ["SQL <query>", "FINISH([answer1, answer2, ...])"],
                                "rules": ["Respond with exactly ONE action per turn — no other text",
                                          "FINISH list must be JSON-loadable (strings inside quotes)",
                                          "FINISH with the answer values themselves (all of them), not a query",
                                          "FINISH([]) when the answer is empty"]}}, indent=2)

    def harness_block(self, paths: list) -> str:
        texts = [Path(p).read_text() for p in paths]
        tools = any(sql.tool_code(t) for t in texts)
        parts = []
        for t in texts:
            s = parse_skill_file(t)
            if tools:                                   # a tool shows its views' names and columns
                s = dict(s, content=self.db.tool_view(s["content"]))
            parts.append(s)
        return render_harness(parts, sql.CODE_NOTE if tools else "")

    @staticmethod
    def _chat(url: str, body: dict, messages: list, deadline: float) -> str:
        body = dict(body, messages=messages)
        for attempt in range(3):
            left = deadline - time.time()
            if left <= 5:
                raise TimeoutError
            try:
                r = requests.post(url, json=body, timeout=min(600, left))
            except requests.Timeout:
                raise TimeoutError from None
            except requests.RequestException:
                time.sleep(2 + 3 * attempt)
                continue
            if r.ok:
                return (r.json()["choices"][0]["message"]["content"] or "").strip()
            if "maximum context length" in r.text or "context length" in r.text:
                raise OverflowError
            time.sleep(2 + 3 * attempt)
        raise ConnectionError(f"server {url} failed")

    def attempt(self, bench: str, job: dict, url: str, sampling: dict, deadline: float) -> dict:
        t0 = time.time()
        task = self.tasks[job["sample_id"]]
        first = self.prompt(task)
        hist = [{"role": "user", "content": first}]
        out = {"job": job["job"], "sample_id": job["sample_id"], "harness": [file_sha(p) for p in job["harness"]],
               "solved": False, "process": None, "status": None, "error": None, "history": hist, "seconds": None}
        blocks = []                                     # (name, views) of every tool
        for p in job["harness"]:
            text = Path(p).read_text()
            if sql.tool_code(text):
                blocks.append((parse_skill_file(text)["name"], sql.tool_code(text)))
        views = sql.prelude(blocks) if blocks else ()
        first_seen = with_harness(first, self.harness_block(job["harness"])) if job["harness"] else first
        body = {"model": sampling["model"], "temperature": sampling["temperature"], "max_tokens": self.turn_tokens,
                "chat_template_kwargs": {"enable_thinking": False}}
        if sampling.get("top_p") is not None:
            body["top_p"] = sampling["top_p"]
        finish, sql_ok, status = None, False, "task limit reached"
        try:
            for _ in range(MAX_TURNS):
                msgs = [{"role": "user", "content": first_seen}] + [
                    {"role": "assistant" if m["role"] == "agent" else "user", "content": m["content"]} for m in hist[1:]]
                reply = self._chat(url, body, msgs, deadline)
                hist.append({"role": "agent", "content": reply})
                kind, arg = sql.parse_action(reply)
                if kind == "finish":
                    finish, status = arg, "completed"
                    break
                if kind == "invalid":
                    status = "agent invalid action"
                    break
                text, ok = self.db.run_sql(arg, views)
                sql_ok = sql_ok or ok
                hist.append({"role": "user", "content": text})
        except TimeoutError:
            out["error"], status = "TIMEOUT", "timeout"
        except OverflowError:
            status = "agent context limit"
        except ConnectionError as e:
            out["error"], status = "NOT_AVAILABLE", str(e)
        out["status"] = status
        if out["error"] is None:
            out["solved"], out["process"] = sql.grade(task, finish, sql_ok)
        out["seconds"] = round(time.time() - t0, 1)
        return out

    # ------------------------------------------------------------ the Improver's evidence

    def trace_problems(self, r: dict) -> list:
        out, st = [], r.get("status")
        if st == "task limit reached":
            out.append("Used all its turns without ever calling FINISH, so nothing it did was graded as an answer.")
        elif st == "agent invalid action":
            last = next((m["content"] for m in reversed(r["history"]) if m["role"] == "agent"), "") or ""
            out.append(f"Its last message was not a valid action, which ended the episode: {last[:220]!r}. Each turn "
                       "must be exactly one plain `SQL <query>` or `FINISH([...])`.")
        elif st == "agent context limit":
            out.append("Ran out of context (the conversation became too long).")
        e = _sql_errors(r)
        if e:
            out.append(f"{e} quer{'y' if e == 1 else 'ies'} failed with an SQL error.")
        if not _queries(r):
            out.append("Answered without running any query.")
        acts = [m["content"] for m in r["history"][1:] if m["role"] == "agent"]
        if len(acts) != len(set(acts)):
            out.append("Sent an identical action more than once instead of changing it after the reply.")
        return out

    def failure_kind(self, r: dict) -> str:
        st = r.get("status")
        if st == "agent invalid action":
            return KINDS[0]
        if st == "task limit reached":
            return KINDS[1]
        if st == "agent context limit":
            return KINDS[2]
        if _sql_errors(r):
            return KINDS[3]
        if not _queries(r):
            return KINDS[4]
        return self.no_trace_problem

    def action_counts(self, r: dict) -> str:
        acts = [m["content"] or "" for m in r["history"][1:] if m["role"] == "agent"]
        fin = any(a.lstrip().startswith("FINISH(") for a in acts)
        return (f"{len(acts)} turns, {_queries(r)} SQL ({_sql_errors(r)} with an error), "
                f"FINISH {'called' if fin else 'never called'}")

    def is_tool(self, text: str) -> bool:
        return bool(sql.tool_code(text))
