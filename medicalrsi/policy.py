"""The dual-role policy: one model, served by vLLM, plays the Solver (attempts tasks with a harness) and the Improver
(writes harnesses from the workflow evidence).

A job is one task with its harness files: {"job", "sample_id", "harness": [paths]} ([] is the initial harness H_0).
Attempts go to <dir>/<tag>_<bench>.jsonl, and a rerun skips the jobs already there."""
from __future__ import annotations

import collections
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from medicalrsi.utils import LOST, load_rows, task_scores, workflow_of


def attempts_file(d: Path, tag: str, bench: str, rerun: bool = False) -> Path:
    return d / f"{tag}_{'rerun_' if rerun else ''}{bench}.jsonl"


class Policy:
    def __init__(self, env, servers, cfg):
        self.env, self.servers, self.rt = env, servers, cfg.runtime

    # ------------------------------------------------------------ the Solver

    def _play(self, jobs: dict, d: Path, tag: str, sampling: dict, concurrency: int, rerun: bool) -> None:
        d.mkdir(parents=True, exist_ok=True)
        todo, files = [], {}
        for b, js in jobs.items():
            if not js:
                continue
            out = attempts_file(d, tag, b, rerun)
            done = {r["job"] for r in load_rows(out)}
            js = list(js)
            random.Random(0).shuffle(js)
            todo += [(b, j) for j in js if j["job"] not in done]
            files[b] = open(out, "a")

        def one(i, b, j):
            return b, self.env.attempt(b, j, self.servers.url(i), sampling, time.time() + self.rt.episode_timeout)

        try:
            with ThreadPoolExecutor(concurrency) as pool:
                for fut in as_completed([pool.submit(one, i, b, j) for i, (b, j) in enumerate(todo)]):
                    b, row = fut.result()
                    files[b].write(json.dumps(row) + "\n")
                    files[b].flush()
        finally:
            for f in files.values():
                f.close()

    def solve(self, jobs: dict, d: Path, tag: str, sampling: dict) -> dict:
        """(bench, job) -> attempt. Lost attempts are played once more."""
        self._play(jobs, d, tag, sampling, self.rt.concurrency, rerun=False)
        again = {}
        for b, js in jobs.items():
            lost = {r["job"] for r in load_rows(attempts_file(d, tag, b)) if r.get("error") in LOST}
            again[b] = [j for j in js if j["job"] in lost]
        if any(again.values()):
            self._play(again, d, tag, sampling, 40, rerun=True)
        res = {}
        for b in jobs:
            for f in (attempts_file(d, tag, b), attempts_file(d, tag, b, rerun=True)):
                for r in load_rows(f):
                    if r.get("error") in LOST and (b, r["job"]) in res:
                        continue
                    res[(b, r["job"])] = r
        return res

    def solver_alone(self, model: str, workflows: set, part: str, d: Path, tag: str) -> dict:
        """The Solver with H_0 at temperature 0 on the `part` tasks of the workflows; (bench, sample) -> attempt."""
        jobs = collections.defaultdict(list)
        for b, s in self.env.split()[part]:
            if (b, workflow_of(s)) in workflows:
                jobs[b].append({"job": f"{tag}|{workflow_of(s)}/-|{s}", "sample_id": s, "harness": []})
        res = self.solve(jobs, d, tag, {"model": model, "temperature": 0.0})
        return {(b, r["sample_id"]): r for (b, _), r in res.items()}

    def solver_with(self, harnesses: dict, model: str, part: str, d: Path, tag: str, temperature: float,
                    top_p=None) -> dict:
        """Every harness {(bench, workflow, name): path} on all `part` tasks of its workflow; (bench, job) -> attempt."""
        jobs = collections.defaultdict(list)
        for b, s in self.env.split()[part]:
            for (bb, t, k), f in harnesses.items():
                if bb == b and workflow_of(s) == t:
                    jobs[b].append({"job": f"{tag}|{t}/{k}|{s}", "sample_id": s, "harness": [str(f)]})
        return self.solve(jobs, d, tag, {"model": model, "temperature": temperature, "top_p": top_p})

    # ------------------------------------------------------------ the Improver

    def improve(self, prompts: dict, model: str, n: int, d: Path, temperature: float, top_p: float, stem: str,
                drop_patient_ids: bool = False, log=None):
        """n candidate harnesses per workflow prompt -> d/harness/<bench>/<workflow>/<stem><k>.md. Returns the
        Improver's samples and {(bench, workflow, name): path} of the candidates that parsed."""
        samples, out = [], {}
        for (b, t), k, text in self.servers.generate(prompts, model, temperature, top_p, n, self.env.write_tokens,
                                                     log or d / "improver.jsonl"):
            p = self.env.parse_harness(text or "")
            names_patient = bool(p and drop_patient_ids and self.env.patient_id.search(p[0]))
            samples.append({"key": [b, t], "k": k, "stem": stem, "text": text, "parsed": p is not None,
                            "failed": text is None,                    # the request failed: nothing was written
                            "names_patient": names_patient, "length": self.env.candidate_length(p) if p else 0})
            if p and not names_patient:
                f = d / "harness" / b / t / f"{stem}{k}.md"
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(p[0])
                out[(b, t, f"{stem}{k}")] = f
        return samples, out


def scores_by_harness(res: dict) -> dict:
    """(bench, workflow, harness name) -> {sample: (outcome, process score)}."""
    per = collections.defaultdict(dict)
    for (b, job), r in res.items():
        v = task_scores([r]).get(r["sample_id"])
        if v is not None:
            t, k = job.split("|")[1].split("/")
            per[(b, t, k)][r["sample_id"]] = v
    return per
