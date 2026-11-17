"""Test-time adaptation and its scores.

The weights stay fixed. On a test workflow whose skill or tool is in memory, the Solver follows it (temperature 0).
On any other workflow, the Solver first attempts its test tasks with H_0, the Improver reads these attempts with their
outcome feedback and the memory of the three most similar training workflows (TF-IDF over the task texts) and writes K
candidate harnesses, and the Solver follows each candidate on the same tasks; a task's score is the mean over the K
candidates. Reported: tasks solved and process score (%) over all, seen and unseen test workflows, with the SD over
1,000 bootstrap resamples of the test tasks."""
from __future__ import annotations

import collections
import json
import random
import re
import statistics as st
from pathlib import Path

from medicalrsi.coevolve import memory_file
from medicalrsi.envs import make_env
from medicalrsi.llm import Servers
from medicalrsi.policy import Policy
from medicalrsi.utils import load_rows, task_scores, workflow_of


def task_text(row: dict) -> str:
    try:
        task = json.loads(row["history"][0]["content"])["task"]
        return f"{task['description']}\nContext: {task['context']}"
    except (ValueError, KeyError, TypeError):
        return (row["history"][0]["content"] or "")[:1500]


def most_similar(alone: dict, test: list, candidates: set, k: int) -> dict:
    """(bench, workflow) -> the k most similar workflows among `candidates`, by TF-IDF over their task texts."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    by = collections.defaultdict(list)
    for b, s in test:
        by[(b, workflow_of(s))].append(s)
    docs = {key: "\n".join(task_text(alone[(key[0], s)]) for s in sorted(ss, key=lambda s: int(s.split("_")[1]))
                           if (key[0], s) in alone) for key, ss in by.items()}
    keys = sorted(docs)
    sim = cosine_similarity(TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform([docs[x] for x in keys]))
    out = {}
    for i, x in enumerate(keys):
        ranked = sorted(((sim[i, j], y) for j, y in enumerate(keys) if y != x and y in candidates), reverse=True)
        out[x] = [y for _, y in ranked[:k]]
    return out


def attempts_in(d: Path, tag: str, benches) -> list:
    """[(bench, attempt)] of <d>/<tag>_<bench>.jsonl, then the reruns."""
    out = []
    for rerun in (False, True):
        for b in sorted(benches):
            out += [(b, r) for r in load_rows(d / f"{tag}_{'rerun_' if rerun else ''}{b}.jsonl")]
    return out


def read_memory(path: Path, base: Path) -> dict:
    """(bench, workflow) -> harness file of a memory file (relative paths from base)."""
    files = {tuple(k.split("/")): memory_file(base, v) for k, v in json.loads(Path(path).read_text()).items()}
    return {k: f for k, f in files.items() if f.exists()}


def test_prompts(env, d: Path, files: dict, k: int):
    """The Improver's prompts for the test workflows without memory: the Solver's attempts with H_0 in <d> and the
    memory of the k most similar workflows. Returns (prompts, references)."""
    test = [tuple(x) for x in env.split()["test"]]
    keep, alone = set(test), {}
    for b, r in attempts_in(d, "alone", env.benches):
        if r.get("history") and (b, r["sample_id"]) in keep and (
                r.get("error") in (None, "None") or (b, r["sample_id"]) not in alone):
            alone[(b, r["sample_id"])] = r
    by = collections.defaultdict(list)
    for (b, s), r in alone.items():
        by[(b, workflow_of(s))].append(r)
    new = [w for w in dict.fromkeys((b, workflow_of(s)) for b, s in test) if w not in files]
    near = most_similar(alone, test, set(files), k)
    prompts = {}
    for w in new:
        refs = [(f"{x} {y}", files[(x, y)].read_text()) for x, y in near[w][:k]]
        if (p := env.reference_prompt(w[0], w[1], by[w], refs)):
            prompts[w] = p
    return prompts, {f"{b}/{w}": [f"{x}/{y}" for x, y in near[(b, w)][:k]] for b, w in new}


def evaluate(cfg, adapter=None, memory=None) -> dict:
    env, T = make_env(cfg), cfg.test
    out = cfg.run / "eval"
    adapter = Path(adapter) if adapter else cfg.run / "adapters" / f"it{cfg.train.rounds}"
    if memory:
        files = read_memory(Path(memory) / "memory.json", Path(memory))
    else:
        files = read_memory(cfg.run / f"it{cfg.train.rounds - 1}" / "memory.json", cfg.run)
    test = [tuple(x) for x in env.split()["test"]]
    workflows = {(b, workflow_of(s)) for b, s in test}
    servers = Servers(cfg, out / "logs")
    env.start(out / "logs")
    servers.start()
    model = f"eval-{cfg.run.name}"
    servers.load_adapter(model, adapter)
    policy = Policy(env, servers, cfg)

    policy.solver_alone(model, workflows, "test", out, "alone")
    policy.solver_with({(b, w, "m"): f for (b, w), f in files.items() if (b, w) in workflows}, model, "test",
                       out / "memory", "memory", 0.0)
    prompts, references = test_prompts(env, out, files, T.references)
    (out / "references.json").write_text(json.dumps(references, indent=1))
    _, written = policy.improve(prompts, model, T.candidates, out / "written", T.temperature, T.top_p, "c",
                                drop_patient_ids=True)
    policy.solver_with(written, model, "test", out / "written", "written", 0.0)
    servers.stop()
    res = scores(env, out, set(files), T.candidates)
    (out / "scores.json").write_text(json.dumps(res, indent=1))
    (out / "scores.txt").write_text(report(res, f"{cfg.env.name} ({cfg.run})") + "\n")
    print(report(res, f"{cfg.env.name} ({cfg.run})"))
    return res


# ---------------------------------------------------------------- scores

def first_scores(d: Path, tag: str, benches) -> dict:
    """(bench, sample) -> {job: (outcome, process)}; a lost attempt yields to its rerun."""
    out = collections.defaultdict(dict)
    for b, x in attempts_in(d, tag, benches):
        v = task_scores([x]).get(x["sample_id"])
        if v is not None and not (x.get("error") == "TIMEOUT" and x["job"] in out[(b, x["sample_id"])]):
            out[(b, x["sample_id"])][x["job"]] = v
    return out


def test_scores(env, d: Path, in_memory: set, k: int) -> dict:
    """(bench, sample) -> (outcome, process score) of MedicalRSI's test protocol."""
    alone = {(b, r["sample_id"]) for b, r in attempts_in(d, "alone", env.benches)}
    rows = collections.defaultdict(list)
    for b, r in attempts_in(d / "written", "written", env.benches):
        rows[(b, r["sample_id"])].append(r)
    cand = re.compile(r"/c(\d+)(?:\||$)")
    out = {}
    for (b, s), rs in rows.items():
        if (b, workflow_of(s)) in in_memory or (b, s) not in alone:
            continue
        last = {}
        for r in rs:
            j = r["job"].split("|")[1]
            m = cand.search(j)
            if m and int(m.group(1)) >= k:
                continue
            if j not in last or r.get("error") in (None, "None"):
                last[j] = r
        vals = [task_scores([r]).get(s) or (0, 0.0) for r in last.values()]
        if vals:
            out[(b, s)] = (st.mean(x[0] for x in vals), st.mean(x[1] for x in vals))
    mem = {key: next(iter(v.values())) for key, v in first_scores(d / "memory", "memory", env.benches).items() if v}
    out.update({key: v for key, v in mem.items() if (key[0], workflow_of(key[1])) in in_memory})
    return out


def bootstrap(per: dict, groups: dict, n: int = 1000, seed: int = 0) -> dict:
    """group -> {solved, solved_sd, process, process_sd, tasks} in %."""
    rng, out = random.Random(seed), {}
    for g, ks in groups.items():
        ps = [(per[k][0], per[k][1] if per[k][1] is not None else float(per[k][0])) for k in ks if k in per]
        if not ps:
            continue
        sd = [100 * st.stdev(st.mean(rng.choices([p[m] for p in ps], k=len(ps))) for _ in range(n)) for m in (0, 1)]
        out[g] = {"solved": 100 * st.mean(p[0] for p in ps), "solved_sd": sd[0],
                  "process": 100 * st.mean(p[1] for p in ps), "process_sd": sd[1], "tasks": len(ps)}
    return out


def scores(env, d: Path, in_memory: set, k: int) -> dict:
    test = [tuple(x) for x in env.split()["test"]]
    unseen = env.heldout()
    groups = {"all": test, "unseen": [x for x in test if (x[0], workflow_of(x[1])) in unseen],
              "seen": [x for x in test if (x[0], workflow_of(x[1])) not in unseen]}
    alone = {key: next(iter(v.values())) for key, v in first_scores(d, "alone", env.benches).items() if v}
    return {"MedicalRSI": bootstrap(test_scores(env, d, in_memory, k), groups),
            "Solver alone": bootstrap(alone, groups)}


def report(res: dict, title: str) -> str:
    lines = [f"{title}: tasks solved and process score (%), mean ± bootstrap SD over test tasks",
             f"{'':28s}{'all':>16s}{'seen':>16s}{'unseen':>16s}"]
    for name in ("MedicalRSI", "Solver alone"):
        for kind in ("solved", "process"):
            c = res[name]
            lines.append(f"{name + ' (' + kind + ')':28s}" + "".join(
                f"{c[g][kind]:9.1f} ± {c[g][kind + '_sd']:3.1f}" if g in c else f"{'-':>16s}" for g in ("all", "seen", "unseen")))
    return "\n".join(lines)
