"""Run configuration. A YAML file per run (configs/) names the model, the environment and the paths; every other value
defaults to the setting of the paper."""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class ModelConfig:
    path: str = "Qwen/Qwen3.5-9B"              # base model (Hugging Face id or local directory)
    name: str = "qwen3.5-9b"                   # served model name
    label: str = "Qwen3.5-9B"                  # the Solver's name in the Improver's prompt
    vllm_args: list = field(default_factory=lambda: ["--reasoning-parser", "qwen3", "--language-model-only"])
    max_lora_rank: int = 64
    encoding: str = "qwen"                     # training encoding: "qwen" chat format or the model's "template"


@dataclass
class EnvConfig:
    name: str = "medagentbench"                # medagentbench | medagentgym
    data_dir: str = "data/medagentbench"
    fhir_url: str = "http://localhost:8080/fhir/"   # MedAgentBench FHIR server
    fhir_snapshot: str = ""                    # MedAgentBench: serve this snapshot when no FHIR server runs
    database: str = ""                         # MedAgentGym: the EHRSQL MIMIC-III SQLite database


@dataclass
class RuntimeConfig:
    gpus: list = field(default_factory=lambda: [0, 1, 2, 3])
    ports: list = field(default_factory=lambda: [8050, 8051, 8052, 8053])   # one vLLM server per GPU
    max_model_len: int = 65536
    gpu_memory_utilization: float = 0.88
    concurrency: int = 150                     # attempts played in parallel
    episode_timeout: int = 600                 # seconds per attempt
    vllm: str = "vllm"
    torchrun: str = "torchrun"
    env: dict = field(default_factory=dict)    # extra environment variables for vLLM and the trainer


@dataclass
class TrainConfig:
    init_adapter: str = ""                     # the warm-start LoRA
    rounds: int = 10
    workflows_per_round: int = 13
    candidates: int = 8                        # G candidate harnesses per workflow
    new_candidates: int = 4                    # for a workflow in memory: candidates written afresh, the rest revise it
    solver_attempts: int = 8                   # attempts with a candidate per task in the Solver's group
    solver_alone_attempts: int = 2             # N attempts with H_0 per task
    temperature: float = 1.0
    top_p: float = 0.95
    memory_margin: float = 0.03                # gain a candidate needs over the stored harness to replace it
    lr: float = 1.0e-5
    steps: int = 2
    clip_low: float = 0.2
    clip_high: float = 0.28
    max_len: int = 32768


@dataclass
class TestConfig:
    candidates: int = 16                       # K candidate harnesses per workflow without memory
    references: int = 3                        # memory of the most similar training workflows
    temperature: float = 0.7
    top_p: float = 0.8


@dataclass
class WarmStartConfig:
    data: str = ""                             # {"prompt", "completion"} per line; built by the teacher if missing
    teacher_model: str = ""                    # an OpenAI-compatible chat model (key in OPENAI_API_KEY)
    teacher_url: str = "https://api.openai.com/v1"
    teacher_reasoning_effort: str = "medium"
    candidates: int = 16                       # teacher harnesses per training workflow
    min_gain: int = 2
    gain_window: int = 3
    keep: int = 12
    keep_no_harm: int = 4
    epochs: int = 3
    lr: float = 1.0e-4
    batch: int = 16
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    max_len: int = 36864


@dataclass
class Config:
    run_dir: str = "runs/medagentbench_qwen3.5-9b"
    model: ModelConfig = field(default_factory=ModelConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    test: TestConfig = field(default_factory=TestConfig)
    warmstart: WarmStartConfig = field(default_factory=WarmStartConfig)

    @property
    def run(self) -> Path:
        return Path(self.run_dir)


SECTIONS = {"ModelConfig": ModelConfig, "EnvConfig": EnvConfig, "RuntimeConfig": RuntimeConfig, "TrainConfig": TrainConfig,
            "TestConfig": TestConfig, "WarmStartConfig": WarmStartConfig}


def _build(cls, data: dict, where: str):
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(names)
    if unknown:
        raise ValueError(f"unknown key(s) in {where}: {sorted(unknown)}")
    kw = {}
    for k, v in data.items():
        sub = SECTIONS.get(str(names[k].type))
        kw[k] = _build(sub, v or {}, f"{where}.{k}") if sub else v
    return cls(**kw)


def load(path, overrides=()) -> Config:
    """The config of a YAML file, with 'section.key=value' overrides."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    for o in overrides:
        key, _, value = o.partition("=")
        node = data
        *parents, last = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[last] = yaml.safe_load(value)
    return _build(Config, data, Path(path).name)


def dump(cfg: Config) -> str:
    return yaml.safe_dump(dataclasses.asdict(cfg), sort_keys=False)
