"""The warm start: harnesses that a stronger LLM writes from the failures of the base agent and that raise its success.

  1 the base model attempts the training tasks with H_0 (temperature 0): the workflow evidence of every training workflow;
  2 the teacher writes `candidates` harnesses per workflow from the Improver's prompt (a harness naming a patient is
    dropped);
  3 the base model attempts every training task of the workflow with each harness (temperature 0);
  4 gain = tasks solved with the harness minus without it. When the best gain is at least min_gain, the harnesses with a
    gain of at least max(min_gain, best - gain_window) are kept (at most keep); otherwise at most keep_no_harm harnesses
    with a gain of at least 0;
  5 supervised fine-tuning of a LoRA adapter on the kept (prompt, harness) pairs (trainer.py warmstart)."""
from __future__ import annotations

import collections
import json
import os
import random
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from medicalrsi.envs import make_env
from medicalrsi.llm import Servers
from medicalrsi.policy import Policy, scores_by_harness
from medicalrsi.utils import task_scores, workflow_of


def teacher(prompts: dict, W, n: int, log: Path) -> list:
    """[(key, k, text)] from an OpenAI-compatible chat API (OPENAI_API_KEY), appended to log; a rerun reuses it."""
    done = {}
    if log.exists():
        for line in open(log):
            d = json.loads(line)
            if d["text"] is not None:
                done[(tuple(d["key"]), d["k"])] = d["text"]
    slots = [(key, k) for key in sorted(prompts) for k in range(n)]
    log.parent.mkdir(parents=True, exist_ok=True)
    headers = {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"}

    def one(slot):
        key, k = slot
        body = {"model": W.teacher_model, "messages": [{"role": "user", "content": prompts[key]}],
                "max_completion_tokens": 16000, "store": False}
        if W.teacher_reasoning_effort:
            body["reasoning_effort"] = W.teacher_reasoning_effort
        for attempt in range(6):
            try:
                r = requests.post(f"{W.teacher_url.rstrip('/')}/chat/completions", headers=headers, json=body, timeout=900)
                if r.ok:
                    return key, k, r.json()["choices"][0]["message"]["content"]
            except requests.RequestException:
                pass
            time.sleep(10 * (attempt + 1))
        return key, k, None

    with ThreadPoolExecutor(32) as pool, open(log, "a") as f:
        for key, k, text in pool.map(one, [s for s in slots if s not in done]):
            done[(key, k)] = text
            f.write(json.dumps({"key": list(key), "k": k, "text": text}) + "\n")
    return [(key, k, done.get((key, k))) for key, k in slots]


def build_data(cfg) -> Path:
    env, W = make_env(cfg), cfg.warmstart
    out = cfg.run / "warmstart_data"
    servers = Servers(cfg, out / "logs")
    env.start(out / "logs")
    servers.start()
    policy = Policy(env, servers, cfg)
    base = cfg.model.name                                      # the base model, without an adapter
    workflows = set(env.train_order())
    alone_rows = policy.solver_alone(base, workflows, "train", out, "alone")
    alone = {k: v for k, r in alone_rows.items() for v in [task_scores([r]).get(k[1])] if v is not None}
    by = collections.defaultdict(list)
    for (b, s), r in alone_rows.items():
        by[(b, workflow_of(s))].append(r)
    prompts = {k: p for k, rs in by.items() if (p := env.improver_prompt(k[0], k[1], rs))}
    harnesses = {}
    for (b, w), k, text in teacher(prompts, W, W.candidates, out / "teacher.jsonl"):
        p = env.parse_harness(text or "")
        if p and not env.patient_id.search(p[0]):
            f = out / "harness" / b / w / f"t{k}.md"
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(p[0])
            harnesses[(b, w, f"t{k}")] = f
    scores = scores_by_harness(policy.solver_with(harnesses, base, "train", out, "teacher", 0.0))
    servers.stop()
    tasks = collections.defaultdict(list)
    for b, s in env.split()["train"]:
        tasks[(b, workflow_of(s))].append(s)
    keep = []
    for (b, w) in sorted(prompts):
        rs = []
        for (bb, ww, k), f in harnesses.items():
            got = scores.get((bb, ww, k), {})
            if (bb, ww) == (b, w) and len(got) == len(tasks[(b, w)]):
                gain = sum(got[s][0] for s in tasks[(b, w)]) - sum(alone[(b, s)][0] for s in tasks[(b, w)] if (b, s) in alone)
                rs.append({"k": k, "gain": gain, "completion": f.read_text(), "prompt": prompts[(b, w)]})
        if not rs:
            continue
        rs.sort(key=lambda r: (-r["gain"], r["k"]))
        best = rs[0]["gain"]
        keep += ([r for r in rs if r["gain"] >= max(W.min_gain, best - W.gain_window)][:W.keep] if best >= W.min_gain
                 else [r for r in rs if r["gain"] >= 0][:W.keep_no_harm])
    random.Random(0).shuffle(keep)
    data = Path(W.data)
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text("".join(json.dumps({"prompt": r["prompt"], "completion": r["completion"]}) + "\n" for r in keep))
    print(f"{len(keep)} warm-start examples from {len(prompts)} workflows -> {data}")
    return data


def train(cfg) -> None:
    W, rt = cfg.warmstart, cfg.runtime
    out = Path(cfg.train.init_adapter)
    if (out / "adapter_model.safetensors").exists():
        print(f"{out} exists")
        return
    if not Path(W.data).exists():
        build_data(cfg)
    env = dict(os.environ, **{k: str(v) for k, v in rt.env.items()}, PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
               CUDA_VISIBLE_DEVICES=",".join(map(str, rt.gpus)))
    cmd = [*rt.torchrun.split(), "--nproc_per_node", str(len(rt.gpus)), "--master_port", "29760", "-m", "medicalrsi.trainer",
           "warmstart", "--model", cfg.model.path, "--data", W.data, "--out", str(out), "--epochs", str(W.epochs),
           "--lr", str(W.lr), "--batch", str(W.batch), "--max_len", str(W.max_len), "--lora_rank", str(W.lora_rank),
           "--lora_alpha", str(W.lora_alpha), "--lora_dropout", str(W.lora_dropout), "--label", cfg.model.label,
           "--encoding", cfg.model.encoding]
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, env=env, check=True)
