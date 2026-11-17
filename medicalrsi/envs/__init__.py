"""The clinical environments. A new benchmark is a subclass of envs.base.Env registered here."""
from __future__ import annotations

import importlib

ENVS = {
    "medagentbench": "medicalrsi.envs.medagentbench:MedAgentBench",
    "medagentgym": "medicalrsi.envs.medagentgym:MedAgentGym",
}


def make_env(cfg):
    if cfg.env.name not in ENVS:
        raise ValueError(f"unknown environment {cfg.env.name!r}; known: {sorted(ENVS)}")
    module, cls = ENVS[cfg.env.name].split(":")
    return getattr(importlib.import_module(module), cls)(cfg)
