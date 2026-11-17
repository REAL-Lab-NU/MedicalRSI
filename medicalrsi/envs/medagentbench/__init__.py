"""MedAgentBench: clinical workflows on a FHIR EHR. The Solver reads a task, queries the FHIR server and places orders.
32 workflows (v1, v2 and our v3) with 30 tasks each: 26 training workflows and 6 unseen ones (data/medagentbench).
The harness is a skill. Each attempt gets its outcome from the version's reference grader and a process score
(graders/)."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml

from medicalrsi.envs.base import Env, render_harness, with_harness
from medicalrsi.envs.medagentbench import protocol as P
from medicalrsi.envs.medagentbench import trace
from medicalrsi.envs.medagentbench.graders.v1 import refsol as _ref1, refsol_partial as _part1
from medicalrsi.envs.medagentbench.graders.v2 import refsol as _ref2, refsol_partial as _part2
from medicalrsi.envs.medagentbench.graders.v3 import refsol as _ref3, refsol_partial as _part3
from medicalrsi.utils import file_sha

VERSIONS = {"MedAgentBench": "v1", "MedAgentBench-v2": "v2", "MedAgentBench-v3": "v3"}
NOW = {"MedAgentBench": "2023-11-13T10:15:00+00:00", "MedAgentBench-v2": "2023-11-07T22:47:00+00:00",
       "MedAgentBench-v3": None}                         # v3: the time stated in the task
GRADERS = {"MedAgentBench": (_ref1, _part1), "MedAgentBench-v2": (_ref2, _part2), "MedAgentBench-v3": (_ref3, _part3)}
for _m in (_ref1, _part1, _ref2, _part2, _ref3, _part3):
    _m.print = lambda *a, **k: None                       # the graders' debugging output

# failure kinds read from the trace, in priority order
KINDS = [("claimed an order in FINISH without an accepted POST", "no POST was accepted in this episode"),
         ("ended by an invalid action", "was not a valid action"),
         ("used all 8 turns without calling FINISH", "without ever calling FINISH"),
         ("ran out of context", "Ran out of context"),
         ("a POST was rejected as invalid", "rejected"),
         ("several actions in one message", "several actions into one message"),
         ("GET requests failed with HTTP errors", "failed with an HTTP error"),
         ("repeated the identical action", "identical action")]


def parse_skill_file(path: Path, name: str) -> dict:
    """{name, description, content} of a skill file, its front matter read as YAML."""
    text = Path(path).read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)", text, re.DOTALL)
    if m:
        meta, body = yaml.safe_load(m.group(1)) or {}, m.group(2).strip()
    else:
        meta, body = {}, text.strip()
    return {"name": meta.get("name", name), "description": meta.get("description", ""), "content": body}


class MedAgentBench(Env):
    name = "medagentbench"
    benches = tuple(VERSIONS)
    improver_template = "improver_medagentbench.txt"
    patient_id = re.compile(r"\bS\d{6,8}\b")             # a patient MRN
    min_scored_tasks = 15

    def __init__(self, cfg):
        super().__init__(cfg)
        self.fhir = cfg.env.fhir_url
        self.funcs = json.loads((self.data_dir / "funcs.json").read_text())
        self.tasks = {b: {s["id"]: s for s in json.loads((self.data_dir / v / "tasks.json").read_text())}
                      for b, v in VERSIONS.items()}
        wf = json.loads((self.data_dir / "workflows.json").read_text())
        self._order = [tuple(x.split("/")) for x in wf["train_order"]]
        self._heldout = {tuple(x.split("/")) for x in wf["heldout"]}

    def load_split(self) -> dict:
        return json.loads((self.data_dir / "split.json").read_text())

    def heldout(self) -> set:
        return set(self._heldout)

    def train_order(self) -> list:
        return list(self._order)

    # ------------------------------------------------------------ the FHIR server

    def fhir_up(self) -> bool:
        try:
            return requests.get(f"{self.fhir}metadata", timeout=10).status_code == 200
        except requests.RequestException:
            return False

    def start(self, log_dir) -> None:
        """Checks the FHIR server; with env.fhir_snapshot set, starts the offline server on the FHIR URL's port."""
        if self.fhir_up():
            return
        snap = self.cfg.env.fhir_snapshot
        if not snap:
            raise RuntimeError(f"no FHIR server at {self.fhir}: start one (README) or set env.fhir_snapshot")
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        port = urlparse(self.fhir).port or 80
        subprocess.Popen([sys.executable, "-m", "medicalrsi.envs.medagentbench.fhir_server", "--snapshot", snap,
                          "--port", str(port)], stdout=open(Path(log_dir) / "fhir_server.log", "a"),
                         stderr=subprocess.STDOUT, start_new_session=True)
        for _ in range(120):
            if self.fhir_up():
                return
            time.sleep(5)
        raise RuntimeError(f"the FHIR server did not start (log in {log_dir})")

    # ------------------------------------------------------------ the Solver

    def harness_block(self, paths: list) -> str:
        files = sorted((f"{Path(p).parent.name}_{Path(p).name}", p) for p in paths)
        return render_harness([parse_skill_file(p, Path(n).stem) for n, p in files])

    def attempt(self, bench: str, job: dict, url: str, sampling: dict, deadline: float) -> dict:
        t0 = time.time()
        sample = self.tasks[bench][job["sample_id"]]
        out = {"job": job["job"], "sample_id": job["sample_id"], "harness": [file_sha(p) for p in job["harness"]],
               "solved": False, "process": None, "status": None, "error": None}
        first = P.task_prompt(self.fhir, self.funcs, sample["context"], sample["instruction"],
                              P.context_now(sample["context"], NOW[bench]))
        output = P.Output(history=[P.Message("user", first)])
        body = {"model": sampling["model"], "temperature": sampling["temperature"], "max_tokens": self.turn_tokens,
                "chat_template_kwargs": {"enable_thinking": False}}
        if sampling.get("top_p") is not None:
            body["top_p"] = sampling["top_p"]
        try:
            first_seen = with_harness(first, self.harness_block(job["harness"])) if job["harness"] else first
            P.play(first, first_seen, self.fhir, url, body, deadline, output)
        except P.Deadline:
            out["error"], out["seconds"] = "TIMEOUT", round(time.time() - t0, 1)
            return out
        except Exception as e:  # noqa: BLE001
            out["error"], out["info"] = "AGENT_FAILED", f"{type(e).__name__}: {e}"[:300]
        out["status"] = output.status
        outcome, process = GRADERS[bench]
        if out["error"] is None and output.result is not None:
            try:
                out["solved"] = getattr(outcome, sample["id"].split("_")[0])(sample, output, self.fhir) is True
            except Exception:  # noqa: BLE001
                out["solved"] = False
        try:
            out["process"] = process.score(sample, output, self.fhir, out["solved"])
        except Exception as e:  # noqa: BLE001
            out["process"] = {"error": f"{type(e).__name__}: {e}"}
        out["history"] = [{"role": m.role, "content": m.content} for m in output.history]
        out["seconds"] = round(time.time() - t0, 1)
        return out

    # ------------------------------------------------------------ the Improver's evidence

    def trace_problems(self, row: dict) -> list:
        return trace.flags(row)

    def failure_kind(self, row: dict) -> str:
        f = " ".join(trace.flags(row))
        return next((name for name, key in KINDS if key in f), self.no_trace_problem)

    def action_counts(self, row: dict) -> str:
        acts = [m["content"] or "" for m in row["history"][1:] if m["role"] == "agent"]
        gets = sum(a.lstrip().startswith("GET") for a in acts)
        posts = sum(a.lstrip().startswith("POST") for a in acts)
        acc = sum(1 for m in row["history"] if m["role"] != "agent" and "POST request accepted" in (m["content"] or ""))
        fin = any(a.lstrip().startswith("FINISH(") for a in acts)
        return f"{len(acts)} turns, {gets} GET, {posts} POST ({acc} accepted), FINISH {'called' if fin else 'never called'}"

    def render_reply(self, text: str) -> str:
        return trace.render_reply(text)
