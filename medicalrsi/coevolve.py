"""Clinical Model-Harness Co-Evolution.

Each round takes the next workflows of the training order and runs, on all their training tasks:
  1 the Solver alone (H_0, temperature 0): the workflow evidence and the baseline of the gain reward;
  2 the Improver writes G candidate harnesses per workflow; a workflow in memory gets revisions of its stored harness
    and new candidates;
  3 the Solver attempts every task with every candidate and the stored harness, and N times with H_0;
  4 the counterfactual gain reward of each candidate and its group-relative advantage;
  5 memory: the best candidate replaces the stored harness when its gain is higher by memory_margin, also when both
    are replayed at temperature 0;
  6 the Solver's group-relative advantages over its attempts on each task;
  7 one joint update of both roles (trainer.py)."""
from __future__ import annotations

import collections
import json
import os
import random
import shutil
import statistics as st
import subprocess
import time
from pathlib import Path

from medicalrsi import config as C
from medicalrsi.envs import make_env
from medicalrsi.envs.base import with_harness
from medicalrsi.llm import Servers
from medicalrsi.policy import Policy, scores_by_harness
from medicalrsi.utils import task_scores, workflow_of, write_jsonl


def length_penalty(n: int) -> float:
    return min(0.1, max(0.0, (n - 1500) * 0.0005))


def gain_reward(env, bench: str, scores: dict, alone: dict):
    """R(H): the gain of u = (outcome + process score) / 2 over the Solver alone, averaged within each case type, case
    types weighted equally. None when too few tasks were scored."""
    per = collections.defaultdict(list)
    for s, (outcome, process) in scores.items():
        if (bench, s) in alone:
            per[env.case_type(bench, s)].append(0.5 * (process - alone[(bench, s)][1]) + 0.5 * (outcome - alone[(bench, s)][0]))
    if sum(len(v) for v in per.values()) < env.min_scored_tasks:
        return None
    return sum(st.mean(v) for v in per.values()) / len(per)


def trajectory(history: list) -> list:
    """An attempt as chat messages, ending with the Solver's last turn."""
    m = [{"role": "assistant" if x["role"] == "agent" else "user", "content": x["content"] or ""} for x in history]
    while m and m[-1]["role"] != "assistant":
        m.pop()
    return m


def load_memory(run: Path, t: int) -> dict:
    """The memory after round t: 'bench/workflow' -> {version, path, gain, round, kind}."""
    f = run / f"it{t}" / "memory.json"
    return json.loads(f.read_text()) if t >= 0 and f.exists() else {}


def memory_file(base: Path, entry: dict) -> Path:
    p = Path(entry["path"])
    return p if p.is_absolute() else base / p


def joint_update(cfg, t: int, adapter: Path, new: Path, batch: Path) -> dict:
    rt, T = cfg.runtime, cfg.train
    env = dict(os.environ, **{k: str(v) for k, v in rt.env.items()}, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               CUDA_VISIBLE_DEVICES=",".join(map(str, rt.gpus)))
    cmd = [*rt.torchrun.split(), "--nproc_per_node", str(len(rt.gpus)), "--master_port", str(29700 + t % 50),
           "-m", "medicalrsi.trainer", "joint", "--model", cfg.model.path, "--batch", str(batch), "--adapter", str(adapter),
           "--out", str(new), "--optim", str(cfg.run / "optimizer"), "--lr", str(T.lr), "--steps", str(T.steps),
           "--clip_low", str(T.clip_low), "--clip_high", str(T.clip_high), "--max_len", str(T.max_len),
           "--encoding", cfg.model.encoding]
    log = batch.with_name("train.log")
    r = subprocess.run(cmd, env=env, stdout=open(log, "a"), stderr=subprocess.STDOUT)
    if r.returncode != 0 or not (new / "train_info.json").exists():
        raise SystemExit(f"training failed: {log}")
    return json.loads((new / "train_info.json").read_text())


def train(cfg) -> None:
    env, T, run = make_env(cfg), cfg.train, cfg.run
    W = run / "adapters"
    if not (W / "it0" / "adapter_model.safetensors").exists():
        if not T.init_adapter:
            raise SystemExit("train.init_adapter is not set")
        (W / "it0").mkdir(parents=True, exist_ok=True)
        for f in ("adapter_config.json", "adapter_model.safetensors"):
            shutil.copy(Path(T.init_adapter) / f, W / "it0" / f)
    (run / "config.yaml").write_text(C.dump(cfg))
    servers = Servers(cfg, run / "logs")
    env.start(run / "logs")
    policy = Policy(env, servers, cfg)
    t = 0
    while (run / f"it{t}" / "done").exists():
        t += 1
    while t < T.rounds:
        t0 = time.time()
        servers.start()
        servers.load_adapter(f"round{t}", W / f"it{t}")
        metrics, n_samples = run_round(cfg, env, policy, t, f"round{t}")
        servers.stop()
        d, new = run / f"it{t}", W / f"it{t + 1}"
        if (new / "train_info.json").exists():
            info = json.loads((new / "train_info.json").read_text())
        elif n_samples:
            info = joint_update(cfg, t, W / f"it{t}", new, d / "batch.jsonl")
        else:
            shutil.copytree(W / f"it{t}", new, dirs_exist_ok=True)
            info = {"samples": 0}
            (new / "train_info.json").write_text(json.dumps(info))
        metrics["train"] = {k: info.get(k) for k in ("samples", "loss", "grad_norm", "seconds")}
        metrics["minutes"] = round((time.time() - t0) / 60, 1)
        with open(run / "metrics.jsonl", "a") as f:
            f.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)
        (d / "done").write_text("ok")
        t += 1


def run_round(cfg, env, policy, t: int, model: str):
    """Steps 1-6 of round t; writes <run>/it<t>/ and returns (metrics, number of training samples)."""
    T, run = cfg.train, cfg.run
    order = env.train_order()
    assert not set(order) & env.heldout()
    d = run / f"it{t}"
    d.mkdir(parents=True, exist_ok=True)
    workflows = {order[(t * T.workflows_per_round + j) % len(order)] for j in range(T.workflows_per_round)}
    state = load_memory(run, t - 1)
    memory = {(b, w): (v["version"], memory_file(run, v)) for k, v in state.items()
              for b, w in [tuple(k.split("/"))] if (b, w) in workflows}

    # 1 the Solver alone: workflow evidence and baseline
    alone_rows = policy.solver_alone(model, workflows, "train", d, "alone")
    alone = {k: v for k, r in alone_rows.items() for v in [task_scores([r]).get(k[1])] if v is not None}
    by = collections.defaultdict(list)
    for (b, s), r in alone_rows.items():
        by[(b, workflow_of(s))].append(r)
    prompts = {k: p for k, rs in by.items() if (p := env.improver_prompt(k[0], k[1], rs))}
    revision, memory_t0 = {}, collections.defaultdict(dict)
    if memory:
        played = policy.solver_with({(b, w, "m"): f for (b, w), (_, f) in memory.items()}, model, "train", d, "memory", 0.0)
        with_memory = collections.defaultdict(list)
        for (b, job), r in played.items():
            w = job.split("|")[1].split("/")[0]
            with_memory[(b, w)].append(r)
            v = task_scores([r]).get(r["sample_id"])
            if v is not None:
                memory_t0[(b, w)][r["sample_id"]] = v
        for (b, w), (n, f) in memory.items():
            p = env.revision_prompt(b, w, with_memory.get((b, w), []), f.read_text(), n)
            if p:
                revision[(b, w)] = p

    # 2 the Improver writes G candidates per workflow
    n_new = min(T.new_candidates, T.candidates) if revision else 0
    samples, candidates = policy.improve({k: p for k, p in prompts.items() if k not in revision}, model, T.candidates, d,
                                         T.temperature, T.top_p, "s")
    for r in samples:
        r["kind"] = "new"
    prompt_of = {(b, w, "new"): p for (b, w), p in prompts.items()}
    if revision:
        rs, cs = policy.improve(revision, model, T.candidates - n_new, d, T.temperature, T.top_p, "s")
        for r in rs:
            r["kind"] = "revision"
        samples += rs
        candidates.update(cs)
        prompt_of.update({(b, w, "revision"): p for (b, w), p in revision.items()})
    if n_new:
        rs, cs = policy.improve({k: prompts[k] for k in revision if k in prompts}, model, n_new, d, T.temperature, T.top_p,
                                "f", log=d / "improver_new.jsonl")
        for r in rs:
            r["kind"] = "new"
        samples += rs
        candidates.update(cs)
    harnesses = dict(candidates)
    harnesses.update({(b, w, "m"): f for (b, w), (_, f) in memory.items()})

    # 3 the Solver attempts every task with every harness, and N times with H_0
    jobs = collections.defaultdict(list)
    for b, s in env.split()["train"]:
        w = workflow_of(s)
        if (b, w) not in workflows:
            continue
        for (bb, ww, k), f in harnesses.items():
            if bb == b and ww == w:
                jobs[b].append({"job": f"round|{w}/{k}|{s}", "sample_id": s, "harness": [str(f)]})
        for g in range(T.solver_alone_attempts):
            jobs[b].append({"job": f"round|{w}/a{g}|{s}", "sample_id": s, "harness": []})
    res = policy.solve(jobs, d, "round", {"model": model, "temperature": T.temperature, "top_p": T.top_p})

    # 4 the Improver: gain rewards and advantages
    scores = scores_by_harness(res)
    for r in samples:
        b, w = r["key"]
        if r.get("failed"):                    # no text was generated: no reward, not in its group or in the update
            r["reward"] = None
            continue
        if not r["parsed"]:
            r["reward"] = -0.5
            continue
        g = gain_reward(env, b, scores.get((b, w, f"{r['stem']}{r['k']}"), {}), alone)
        r["reward"] = None if g is None else g - length_penalty(r["length"])
    groups = collections.defaultdict(list)
    for r in samples:
        if r["reward"] is not None:
            groups[(*r["key"], r["kind"])].append(r)
    improver_batch = []
    for key, rs in groups.items():
        mu = st.mean(r["reward"] for r in rs)
        for r in rs:
            r["adv"] = r["reward"] - mu
            if abs(r["adv"]) > 1e-9:
                improver_batch.append({"prompt": prompt_of[key], "completion": r["text"], "adv": r["adv"]})
    write_jsonl(d / "improver_samples.jsonl", samples)

    # 5 memory
    log, best_of = [], {}
    for (b, w) in sorted(workflows):
        cs = [r for r in samples if tuple(r["key"]) == (b, w) and r["reward"] is not None and r["parsed"]]
        if not cs:
            continue
        best = max(cs, key=lambda r: (r["reward"], -r["k"]))
        cur = memory.get((b, w))
        stored = gain_reward(env, b, scores.get((b, w, "m"), {}), alone) if cur else None
        ok = best["reward"] > 0 if cur is None else (stored is not None and best["reward"] >= stored + T.memory_margin)
        log.append({"workflow": f"{b}/{w}", "version": cur[0] if cur else 0, "stored_gain": stored,
                    "best_gain": best["reward"], "replace": bool(ok), "best_kind": best["kind"]})
        best_of[(b, w)] = (candidates[(b, w, f"{best['stem']}{best['k']}")], best["length"], best["kind"])
    replay = {(*x["workflow"].split("/"), "c"): best_of[tuple(x["workflow"].split("/"))][0] for x in log if x["replace"]}
    confirm = collections.defaultdict(dict)
    for (b, job), r in (policy.solver_with(replay, model, "train", d / "confirm", "confirm", 0.0) if replay else {}).items():
        v = task_scores([r]).get(r["sample_id"])
        if v is not None:
            confirm[(b, job.split("|")[1].split("/")[0])][r["sample_id"]] = v
    for x in log:
        if not x["replace"]:
            continue
        b, w = x["workflow"].split("/")
        g = gain_reward(env, b, confirm.get((b, w), {}), alone)
        g = None if g is None else g - length_penalty(best_of[(b, w)][1])
        x["confirm_gain"] = g
        if x["version"] == 0:
            x["replace"] = g is not None and g > 0
            continue
        g0 = gain_reward(env, b, memory_t0.get((b, w), {}), alone)
        g0 = None if g0 is None else g0 - length_penalty(env.stored_length(memory[(b, w)][1].read_text()))
        x["stored_gain_t0"] = g0
        x["replace"] = g is not None and g0 is not None and g >= g0 + T.memory_margin
    for x in log:
        if x["replace"]:
            b, w = x["workflow"].split("/")
            src, _, kind = best_of[(b, w)]
            cur = memory.get((b, w))
            n = (cur[0] if cur else 0) + 1
            rel = Path("memory") / b / w / f"v{n}.md"
            (run / rel).parent.mkdir(parents=True, exist_ok=True)
            (run / rel).write_text(src.read_text())
            state[f"{b}/{w}"] = {"version": n, "path": str(rel), "gain": x["confirm_gain"], "round": t, "kind": kind}
    (d / "memory.json").write_text(json.dumps(state, indent=1))
    (d / "memory_log.json").write_text(json.dumps(log, indent=1))

    # 6 the Solver: every attempt as it was played, against the other attempts on the same task
    blocks, with_h, without_h = {}, collections.defaultdict(list), collections.defaultdict(list)
    for (b, job), r in res.items():
        msgs = trajectory(r.get("history") or [])
        if r.get("error") not in (None, "None") or not msgs:
            continue
        w, k = job.split("|")[1].split("/")
        if (b, w, k) not in harnesses:
            without_h[(b, r["sample_id"])].append((int(bool(r["solved"])), msgs, f"{b}|{job}"))
            continue
        f = harnesses[(b, w, k)]
        if f not in blocks:
            blocks[f] = env.harness_block([f])
        msgs[0] = dict(msgs[0], content=with_harness(msgs[0]["content"], blocks[f]))
        with_h[(b, r["sample_id"])].append((int(bool(r["solved"])), msgs, f"{b}|{job}"))
    solver_batch, mixed = [], 0
    for (b, s), xs in sorted(with_h.items()):
        group = random.Random(f"{t}|{b}|{s}").sample(sorted(xs, key=lambda x: x[2]), min(T.solver_attempts, len(xs)))
        group += sorted(without_h.get((b, s), []), key=lambda x: x[2])
        rewards = [x[0] for x in group]
        if max(rewards) == min(rewards):
            continue
        mixed += 1
        mu = st.mean(rewards)
        solver_batch += [{"job": job, "messages": msgs, "adv": r - mu} for r, msgs, job in group]

    write_jsonl(d / "batch.jsonl", [dict(r, role="improver") for r in improver_batch]
                + [dict(r, role="solver") for r in solver_batch])
    gains = [r["reward"] for r in samples if r["reward"] is not None]
    metrics = {"round": t, "workflows": sorted(f"{b}/{w}" for b, w in workflows),
               "solver_alone": round(st.mean(v[0] for v in alone.values()), 4) if alone else None,
               "improver_gain": round(st.mean(gains), 4) if gains else None,
               "unparsed": sum(not r["parsed"] and not r.get("failed") for r in samples),
               "failed_generations": sum(bool(r.get("failed")) for r in samples),
               "memory_updates": sum(x["replace"] for x in log),
               "improver_samples": len(improver_batch), "solver_samples": len(solver_batch), "solver_tasks": mixed}
    return metrics, len(improver_batch) + len(solver_batch)
