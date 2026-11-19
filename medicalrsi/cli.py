"""python -m medicalrsi <command> --config configs/<run>.yaml

  warmstart   the warm start: teacher harnesses (if warmstart.data is missing), then SFT into train.init_adapter
  train       Clinical Model-Harness Co-Evolution for train.rounds rounds (resumable), then the test
  evaluate    test-time adaptation with the run's last adapter and memory, or a given --adapter and --memory
  score       the scores of a finished evaluation

Config values can be overridden: --set train.rounds=2 --set runtime.gpus=[0,1]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from medicalrsi import config as C


def main() -> None:
    ap = argparse.ArgumentParser(prog="medicalrsi", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["warmstart", "train", "evaluate", "score"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--adapter", help="a LoRA adapter directory (evaluate)")
    ap.add_argument("--memory", help="a memory directory with memory.json (evaluate, score)")
    a = ap.parse_args()
    cfg = C.load(a.config, a.set)
    if a.command == "warmstart":
        from medicalrsi import warmstart
        warmstart.train(cfg)
    elif a.command == "train":
        from medicalrsi import coevolve, evaluate
        coevolve.train(cfg)
        evaluate.evaluate(cfg)
    elif a.command == "evaluate":
        from medicalrsi import evaluate
        evaluate.evaluate(cfg, a.adapter, a.memory)
    else:
        from medicalrsi import evaluate
        from medicalrsi.envs import make_env
        mark = evaluate.read_mark(cfg.run / "eval")          # the memory of the evaluation that is in <run>/eval
        if a.memory or not mark:
            f = Path(a.memory) / "memory.json" if a.memory else cfg.run / f"it{cfg.train.rounds - 1}" / "memory.json"
            in_memory = {tuple(k.split("/")) for k in json.loads(f.read_text())}
        else:
            in_memory = {tuple(k.split("/")) for k in mark["workflows_in_memory"]}
        if mark and {"/".join(k) for k in in_memory} != set(mark["workflows_in_memory"]):
            raise SystemExit(f"{cfg.run / 'eval'} holds an evaluation with another memory ({mark['memory']})")
        res = evaluate.scores(make_env(cfg), cfg.run / "eval", in_memory, cfg.test.candidates)
        print(evaluate.report(res, f"{cfg.env.name} ({cfg.run})"))


if __name__ == "__main__":
    main()
