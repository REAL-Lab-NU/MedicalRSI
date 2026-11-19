"""vLLM servers, one per GPU, serving the base model with the current LoRA adapter. They serve both roles during a
round and are stopped for the update."""
from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests


class Servers:
    def __init__(self, cfg, log_dir):
        self.cfg, self.rt, self.model = cfg, cfg.runtime, cfg.model
        self.ports = [int(p) for p in self.rt.ports]
        self.gpus = [str(g) for g in self.rt.gpus]
        if len(self.ports) != len(self.gpus):
            raise ValueError("runtime.ports and runtime.gpus must have the same length")
        self.log_dir = Path(log_dir)
        self.pid_file = self.log_dir / "vllm_pids.json"

    def url(self, i: int) -> str:
        return f"http://127.0.0.1:{self.ports[i % len(self.ports)]}/v1/chat/completions"

    def up(self, timeout: float = 900) -> bool:
        """True when every server answers; False after the timeout or as soon as a server of this run has exited."""
        t0 = time.time()
        while True:
            ok = 0
            for p in self.ports:
                try:
                    ok += requests.get(f"http://127.0.0.1:{p}/v1/models", timeout=3).ok
                except requests.RequestException:
                    pass
            if ok == len(self.ports):
                return True
            if time.time() - t0 > timeout or (self.pid_file.exists() and len(self._pids()) < len(self.ports)):
                return False
            time.sleep(5)

    def _pids(self) -> list[int]:
        if not self.pid_file.exists():
            return []
        return [p for p in json.loads(self.pid_file.read_text()) if _alive(p)]

    def _launch(self, attempt: int) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        pids = []
        for i, (gpu, port) in enumerate(zip(self.gpus, self.ports)):
            env = dict(os.environ, **{k: str(v) for k, v in self.rt.env.items()}, CUDA_VISIBLE_DEVICES=gpu,
                       VLLM_PORT=str(20000 + 1000 * attempt + 100 * i),     # separate internal ports per server
                       VLLM_USE_V2_MODEL_RUNNER="0", VLLM_ALLOW_RUNTIME_LORA_UPDATING="True")
            cmd = [*shlex.split(self.rt.vllm), "serve", self.model.path, "--served-model-name", self.model.name,
                   "--host", "127.0.0.1", "--port", str(port), "--max-model-len", str(self.rt.max_model_len),
                   "--gpu-memory-utilization", str(self.rt.gpu_memory_utilization), "--enable-prefix-caching",
                   *self.model.vllm_args, "--enable-lora", "--max-lora-rank", str(self.model.max_lora_rank),
                   "--max-loras", "2", "--max-cpu-loras", "4"]
            p = subprocess.Popen(cmd, env=env, stdout=open(self.log_dir / f"vllm_{port}.log", "a"),
                                 stderr=subprocess.STDOUT, start_new_session=True)
            pids.append(p.pid)
        self.pid_file.write_text(json.dumps(pids))

    def start(self) -> None:
        if self.up(timeout=5):
            return
        for attempt in range(3):
            if not self._pids():
                self._launch(attempt)
            if self.up():
                return
            self.stop()
        raise RuntimeError(f"vLLM servers did not come up (logs in {self.log_dir})")

    def stop(self) -> None:
        """Stops the servers this run started and waits until their GPUs are free."""
        for pid in self._pids():
            _signal(pid, signal.SIGTERM)
        for i in range(120):
            if not self._pids() and _gpus_free(self.gpus):
                break
            if i == 20:
                for pid in self._pids():
                    _signal(pid, signal.SIGKILL)
            time.sleep(3)
        else:
            raise RuntimeError("the GPUs were not freed after stopping the vLLM servers")
        self.pid_file.unlink(missing_ok=True)

    def load_adapter(self, name: str, path) -> None:
        for p in self.ports:
            r = requests.post(f"http://127.0.0.1:{p}/v1/load_lora_adapter",
                              json={"lora_name": name, "lora_path": str(Path(path).resolve())}, timeout=300)
            if not r.ok and "already" not in r.text:
                raise RuntimeError(f"loading adapter {name} failed on port {p}: {r.text[:300]}")

    def generate(self, prompts: dict, model: str, temperature: float, top_p: float, n: int, max_tokens: int,
                 log_path) -> list[tuple]:
        """[(key, k, text)] for n completions of every prompt, appended to log_path; a rerun reuses the logged ones.
        A request that fails three times returns text None; it is not logged, so a rerun asks again."""
        slots = [(key, k) for key in sorted(prompts) for k in range(n)]
        done = {}
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            for line in open(log_path):
                d = json.loads(line)
                if d["text"] is not None:
                    done[(tuple(d["key"]), d["k"])] = d["text"]
        todo = [(i, s) for i, s in enumerate(slots) if s not in done]

        def one(item):
            i, (key, k) = item
            for _ in range(3):
                try:
                    r = requests.post(self.url(i), timeout=900, json={
                        "model": model, "messages": [{"role": "user", "content": prompts[key]}],
                        "temperature": temperature, "top_p": top_p, "max_tokens": max_tokens,
                        "chat_template_kwargs": {"enable_thinking": False}}).json()
                    return key, k, r["choices"][0]["message"]["content"]
                except Exception:  # noqa: BLE001
                    time.sleep(5)
            return key, k, None

        failed = 0
        with ThreadPoolExecutor(64) as pool, open(log_path, "a") as f:
            for key, k, text in pool.map(one, todo):
                if text is None:
                    failed += 1
                    continue
                done[(key, k)] = text
                f.write(json.dumps({"key": list(key), "k": k, "text": text}) + "\n")
        if failed:
            print(f"{failed} of {len(slots)} generation requests failed after three tries; they are left out, and a rerun "
                  f"asks again")
        return [(key, k, done.get((key, k))) for key, k in slots]


def _signal(pid: int, sig) -> None:
    try:
        os.killpg(pid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def _alive(pid: int) -> bool:
    try:
        os.waitpid(pid, os.WNOHANG)                       # reaps a server that has exited
    except ChildProcessError:
        pass
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False
    except OSError:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True


def _gpus_free(gpus: list[str], limit_mib: int = 2000) -> bool:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True).stdout.split()
    except (OSError, subprocess.CalledProcessError):
        return True
    return all(int(out[int(g)]) < limit_mib for g in gpus)
