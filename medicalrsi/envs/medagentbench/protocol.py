"""One MedAgentBench attempt: the Solver sends one action per turn (GET a FHIR URL, POST an order with its JSON payload,
or FINISH with its answers), at most 8 turns. GETs go to the FHIR server; POSTs are checked and recorded for the
graders. The task prompt and the action parsing follow the MedAgentBench task server (graders/LICENSE)."""
from __future__ import annotations

import json
import random
import re
import time
from dataclasses import dataclass, field

import requests

MAX_ROUNDS = 8
DEFAULT_NOW = "2023-11-13T10:15:00+00:00"
REPLY_TAIL = ". Please call FINISH if you have got answers for all the questions and finished all the requested tasks"
ACTION_PREFIXES = ("GET", "POST", "FINISH(")


@dataclass
class Message:
    role: str           # "user" (the environment) or "agent"
    content: str


@dataclass
class Output:
    """What the graders read: the history, the FINISH answer (the text inside FINISH(...)) and how the episode ended."""
    history: list = field(default_factory=list)
    result: object = None
    status: str = "running"


class ContextLimit(Exception):
    pass


class AgentFailed(Exception):
    pass


class Deadline(Exception):
    pass


def context_now(context: str, now: str | None) -> str:
    """The current time shown to the agent: fixed per benchmark version, or (v3) the task's own "It's <time> now"."""
    if now is not None:
        return now
    m = re.search(r"It'?s\s+(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?[+-]\d{2}:\d{2})\s*now", context or "", re.IGNORECASE)
    return m.group(1) if m else DEFAULT_NOW


def task_prompt(api_base: str, functions: list, context: str, question: str, now: str) -> str:
    return json.dumps({
        "phase": "task_execution",
        "task": {
            "description": question,
            "context": (context + "\n" if context else "") + f"Current time: {now}",
        },
        "behavioral_skills": "",
        "api": {
            "base_url": api_base,
            "functions": functions,
        },
        "response_format": {
            "type": "api_action",
            "options": [
                "GET url?param_name1=param_value1&param_name2=param_value2",
                "POST url\n{json_payload}",
                "FINISH([answer1, answer2, ...])",
            ],
            "rules": [
                "Respond with exactly ONE action per turn — no other text",
                "FINISH list must be JSON-loadable (strings inside quotes)",
            ],
        },
    }, indent=2)


def lenient_action(r: str) -> str:
    """One action from a response that wraps it in JSON (e.g. {"action": "GET ..."}) or puts text before it;
    a response that starts with an action is returned unchanged."""
    if r.startswith(ACTION_PREFIXES):
        return r
    try:
        obj = json.loads(r)
    except Exception:  # noqa: BLE001
        obj = None
    if isinstance(obj, dict):
        for key in ("action", "api_action", "response", "next_action"):
            v = obj.get(key)
            if isinstance(v, str) and v.strip().startswith(ACTION_PREFIXES):
                v = v.strip()
                body = obj.get("payload", obj.get("body", obj.get("json_payload")))
                if v.startswith("POST") and "\n" not in v and isinstance(body, (dict, list)):
                    v = v + "\n" + json.dumps(body)
                return v
    lines = r.split("\n")
    for i, line in enumerate(lines):
        t = line.strip()
        if t.startswith(("GET", "FINISH(")):
            return t
        if t.startswith("POST"):
            return "\n".join([t] + lines[i + 1:]).strip()
    return r


def send_get(url: str) -> dict:
    try:
        response = requests.get(url)
        response.raise_for_status()
        return {"status_code": response.status_code,
                "data": response.json() if response.headers.get("Content-Type") == "application/json" else response.text}
    except Exception as e:  # noqa: BLE001
        return {"error": str(e)}


def _context_limit(text: str) -> bool:
    t = text.lower()
    return (any(w in t for w in ("prompt", "context", "tokens"))
            and any(w in t for w in ("limit", "exceed", "max", "long", "much", "many", "reach", "over", "up", "beyond")))


def _retry_delay(attempt: int, base: float = 5.0, cap: float = 60.0) -> float:
    delay = min(base * (2 ** attempt), cap)
    return max(1.0, delay + delay * 0.2 * (random.random() - 0.5))


def call_model(url: str, body: dict, messages: list, deadline: float, retries: int = 3) -> str:
    """The agent's reply: the message content, or (empty content) the whole response as text."""
    for attempt in range(retries):
        left = deadline - time.time()
        if left <= 0:
            raise Deadline
        try:
            resp = requests.post(url, json=dict(body, messages=messages), timeout=min(600.0, left))
            if resp.status_code == 429:
                time.sleep(_retry_delay(attempt))
                continue
            if resp.status_code != 200:
                if _context_limit(resp.text):
                    raise ContextLimit(resp.text)
                raise RuntimeError(f"Invalid status code {resp.status_code}:\n\n{resp.text}")
        except ContextLimit:
            raise
        except Exception as e:  # noqa: BLE001
            if time.time() >= deadline:
                raise Deadline from e
            if attempt == retries - 1:
                raise AgentFailed(str(e)) from e
            time.sleep(_retry_delay(attempt))
            continue
        data = resp.json()
        if isinstance(data, dict) and data.get("choices"):
            message = data["choices"][0].get("message", {})
            if "content" in message and message["content"]:
                return message["content"]
        return str(data)
    raise AgentFailed("Failed.")


def play(first: str, agent_first: str, fhir: str, url: str, body: dict, deadline: float, out: Output) -> Output:
    """Plays one episode into `out` (its history starts with the task prompt). The agent sees `agent_first` (the task
    prompt with its harness) as its first message; the history keeps the prompt without it."""
    try:
        for _ in range(MAX_ROUNDS):
            msgs = [{"role": "user", "content": agent_first}] + [
                {"role": "assistant" if m.role == "agent" else "user", "content": m.content} for m in out.history[1:]]
            try:
                content = call_model(url, body, msgs, deadline)
            except ContextLimit:
                out.history.append(Message("agent", "agent context limit"))
                out.status = "agent context limit"
                return out
            r = content.strip().replace("```tool_code", "").replace("```", "").strip()
            r = lenient_action(r)
            out.history.append(Message("agent", r))         # the graders read POSTs from the history
            if r.startswith("GET"):
                res = send_get(r[3:].strip() + "&_format=json")
                if "data" in res:
                    out.history.append(Message("user", f"Here is the response from the GET request:\n{res['data']}{REPLY_TAIL}"))
                else:
                    out.history.append(Message("user", f"Error in sending the GET request: {res['error']}"))
            elif r.startswith("POST"):
                try:
                    json.loads("\n".join(r.split("\n")[1:]))
                except Exception:  # noqa: BLE001
                    out.history.append(Message("user", "Invalid POST request"))
                else:
                    out.history.append(Message("user", "POST request accepted and executed successfully" + REPLY_TAIL))
            elif r.startswith("FINISH("):
                out.result, out.status = r[len("FINISH("):-1], "completed"
                return out
            else:
                out.status = "agent invalid action"
                return out
    except (AgentFailed, Deadline):
        raise
    except Exception as e:  # noqa: BLE001
        out.result, out.status = {"error": str(e)}, "task error"
        return out
    out.status = "task limit reached"
    return out
