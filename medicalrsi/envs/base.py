"""A clinical environment: its workflows, the Solver's attempts, the harness format and the Improver's prompts.

Subclasses implement load_split, heldout, train_order, attempt, harness_block and the trace readers (failure_kind,
trace_problems, action_counts); everything else is shared. A new benchmark is a subclass registered in envs/__init__.py.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

from medicalrsi.utils import clean_success, finished, hkey

PROMPTS = Path(__file__).resolve().parents[1] / "prompts"

MAX_FAILED, MAX_SOLVED, MAX_SOLVED_ONLY = 5, 2, 3
CONTRAST = ("\nPASSED episodes are complete successes: compare them with the FAILED ones and find the step where they "
            "diverge; keep what the passed episodes did.")
ALL_SOLVED = ("\nThe student solved every reference task of this family without a skill: a skill can keep it that way or "
              "make it worse.")
REVISE = ("- Revise the current skill into a complete new version: keep the steps the passed episodes followed, fix or add "
          "steps for what the failed episodes got wrong, and drop steps that misled the student. If no change would "
          "help, output the current skill unchanged.\n")
REVISE_TOOLS = ("- Revise the current skill or tool into a complete new version: keep what the passed episodes did, fix or "
                "add what the failed episodes got wrong (for a tool: its code or its documentation), and drop what misled "
                "the student. A skill may become a tool, or a tool a skill, if that helps the student more. If no change "
                "would help, output it unchanged.\n")
REFERENCES = """SKILLS FROM THE STUDENT'S LIBRARY for the {m} most similar question templates it has trained on{own}. Reuse the
steps that apply to this template, correct what the episodes show to be wrong or missing, and drop what does not apply:

{skills}

"""
HARNESS_HEADER = ("---\n**Behavioral skills:** before each action, scan the skill descriptions below. If a skill's 'When to "
                  "use' matches your current task or the action you are about to take, follow its guidance. Skip skills "
                  "that do not match.\n")


def template(name: str) -> str:
    text = (PROMPTS / name).read_text()
    return text[:-1] if text.endswith("\n") else text


def render_harness(parts: list, note: str = "") -> str:
    """The harness block of the Solver's first prompt, from parsed harness files [{name, description, content}]."""
    blocks = []
    for s in parts:
        when = f"*When to use: {s['description']}*\n" if s["description"] else ""
        blocks.append(f"### {s['name']}\n{when}\n{s['content']}")
    return HARNESS_HEADER + note + "\n\n".join(blocks)


def with_harness(first: str, block: str) -> str:
    """The Solver's first prompt with the harness block in its behavioral_skills field."""
    try:
        d = json.loads(first)
        for k in ("skill_instruction", "selected_skills", "skill_documentation"):
            d.pop(k, None)
        d["behavioral_skills"] = block
        return json.dumps(d, indent=2)
    except (json.JSONDecodeError, TypeError, AttributeError):
        return first + "\n\n" + block


def clip(text: str, n: int = 1200) -> str:
    text = text or ""
    return text[:n] + (f" ...[{len(text) - n} more chars]" if len(text) > n else "")


def process_score(r: dict) -> float:
    return float(((r.get("process") or {}).get("score")) or 0.0)


class Env:
    name = ""
    benches: tuple = ()
    improver_template = ""             # the Improver's prompt template (medicalrsi/prompts)
    has_tools = False                  # the Improver may write a tool (code) as well as a skill (text)
    code_block = None                  # the regex of a tool's code (group 1)
    code_chars = 2000                  # tool code that does not count toward the harness length
    body_chars = 1200                  # the harness length the Improver is asked to keep to
    write_tokens = 1200                # the Improver's output budget
    turn_tokens = 512                  # the Solver's output budget per turn
    min_scored_tasks = 15              # tasks a candidate needs to be scored on to get a gain reward
    patient_id = re.compile(r"(?!x)x")  # a test-time harness that names a patient is dropped
    no_trace_problem = "no problem visible in the trace (the decision, a value or the payload was wrong)"

    def __init__(self, cfg):
        self.cfg = cfg
        self.solver_name = cfg.model.label
        self.data_dir = Path(cfg.env.data_dir)
        self.improver_text = template(self.improver_template)
        self._split = None

    # ------------------------------------------------------------ workflows

    def load_split(self) -> dict:
        raise NotImplementedError

    def split(self) -> dict:
        """{"train": [[bench, sample]], "test": [...], "branch": {"bench|sample": case type}}"""
        if self._split is None:
            self._split = self.load_split()
        return self._split

    def heldout(self) -> set:
        """The unseen workflows (bench, workflow), never trained on."""
        raise NotImplementedError

    def train_order(self) -> list:
        """The training workflows in the fixed order the rounds visit them."""
        raise NotImplementedError

    def case_type(self, bench: str, sample: str) -> str:
        g = self.split()["branch"].get(f"{bench}|{sample}", "read-only")
        return g if g in ("order expected", "no order expected") else "read-only"

    # ------------------------------------------------------------ the Solver

    def start(self, log_dir) -> None:
        """Starts the services the attempts need."""

    def attempt(self, bench: str, job: dict, url: str, sampling: dict, deadline: float) -> dict:
        raise NotImplementedError

    def harness_block(self, paths: list) -> str:
        """The harness block the Solver sees for these harness files."""
        raise NotImplementedError

    # ------------------------------------------------------------ harness files

    def parse_harness(self, text: str):
        """(file text, name, length) of the harness in an Improver answer, or None."""
        t = text.strip()
        if self.has_tools:
            m = re.match(r"^```(?:markdown|md)?[ \t]*\n(.*)\n```$", t, re.S)
        else:
            m = re.search(r"```(?:markdown|md)?\s*\n(.*?)```", t, re.S)
        if m:
            t = m.group(1).strip()
        if not t.startswith("---"):
            return None
        parts = t.split("---", 2)
        if len(parts) < 3:
            return None
        fm, body = parts[1], parts[2].strip()
        name = re.search(r"^name:\s*(\S+)", fm, re.M)
        desc = re.search(r"^description:\s*(.+)$", fm, re.M)
        if not name or not desc:
            return None
        n = len(body)
        if self.has_tools:
            c = self.code_block.search(body)
            if c:
                n = len(body) - len(c.group(0)) + max(0, len(c.group(1)) - self.code_chars)
            elif re.search(r"```(?:python|sql)\b", body, re.I):
                return None
        return t + "\n", name.group(1).strip().strip('"'), n

    def candidate_length(self, parsed) -> int:
        return parsed[2] if self.has_tools else len(parsed[0].split("---", 2)[2])

    def stored_length(self, text: str) -> int:
        p = self.parse_harness(text)
        return p[2] if p else (len(text.split("---", 2)[2]) if text.count("---") >= 2 else len(text))

    def is_tool(self, text: str) -> bool:
        return False

    # ------------------------------------------------------------ the Improver's prompts

    def failure_kind(self, r: dict) -> str:
        raise NotImplementedError

    def trace_problems(self, r: dict) -> list:
        raise NotImplementedError

    def action_counts(self, r: dict) -> str:
        raise NotImplementedError

    def render_reply(self, text: str) -> str:
        return clip(text)

    def render_attempt(self, i: int, r: dict) -> str:
        h = r["history"]
        ok = bool(r["solved"])
        lines = [f"--- Episode {i}: {'PASSED (full credit)' if ok else f'FAILED (partial score {process_score(r):.2f})'} ---"]
        try:
            task = json.loads(h[0]["content"])["task"]
            lines.append(f"Task: {task['description']}\nContext: {task['context']}")
        except Exception:  # noqa: BLE001
            lines.append("Task: " + (h[0]["content"] or "")[:1500])
        for m in h[1:]:
            lines.append(f"STUDENT: {m['content']}" if m["role"] == "agent" else f"ENVIRONMENT: {self.render_reply(m['content'])}")
        lines.append(f"How it ended: {r['status']}. Actions: {self.action_counts(r)}.")
        if not ok:
            fl = self.trace_problems(r)
            lines.append("Problems detected from the trace:\n" + "\n".join(f" - {x}" for x in fl) if fl else
                         "Problems detected from the trace: none (the decision, a value or the payload must have been wrong).")
        text = "\n".join(lines)
        if len(text) > 14000:
            text = text[:7000] + f"\n...[{len(text) - 14000} characters of this episode omitted]...\n" + text[-7000:]
        return text

    def select_evidence(self, attempts: list) -> tuple:
        """(failed, solved) attempts of the workflow evidence: failures spread over failure kinds, then successes."""
        rows = sorted(finished(attempts), key=hkey)
        failed = [r for r in rows if not r["solved"]]
        solved = [r for r in rows if clean_success(r)] or [r for r in rows if r["solved"]]
        kinds = collections.defaultdict(list)
        for r in failed:
            kinds[self.failure_kind(r)].append(r)
        order = sorted(kinds.values(), key=lambda g: (-len(g), hkey(g[0])))
        F = []
        while len(F) < MAX_FAILED and any(order):
            for g in order:
                if g and len(F) < MAX_FAILED:
                    F.append(g.pop(0))
        return F, (solved[:MAX_SOLVED] if F else solved[:MAX_SOLVED_ONLY])

    def improver_prompt(self, bench: str, workflow: str, attempts: list):
        """The Improver's prompt with the workflow evidence of the Solver's attempts; None without attempts."""
        rows = finished(attempts)
        if not rows:
            return None
        F, P = self.select_evidence(rows)
        n, ok = len(rows), sum(bool(r["solved"]) for r in rows)
        failed = [r for r in rows if not r["solved"]]
        kc = collections.Counter(self.failure_kind(r) for r in failed)
        stats = (f"Without a skill the student fully solved {ok} of these {n} tasks (mean partial score "
                 f"{sum(process_score(r) for r in rows) / n:.2f}).")
        if failed:
            stats += (f" Failure kinds read from the traces of all {len(failed)} failed episodes: "
                      + "; ".join(f"{k}: {v}" for k, v in kc.most_common()) + f". Shown below: {len(F)} failed and "
                      f"{len(P)} passed episode(s).")
        eps = "\n\n".join(self.render_attempt(i + 1, r) for i, r in enumerate(F + P))
        return self.improver_text.format(student=self.solver_name, task=f"{bench} {workflow}", n=n, stats=stats,
                                         contrast=(CONTRAST if P else "") if F else ALL_SOLVED,
                                         prompt=(F or P)[0]["history"][0]["content"], episodes=eps, body=self.body_chars)

    def revision_prompt(self, bench: str, workflow: str, attempts: list, harness: str, version: int):
        """The Improver's prompt to revise the workflow's harness in memory from the attempts that used it."""
        base = self.improver_prompt(bench, workflow, attempts)
        if base is None:
            return None
        what = "skill or tool" if self.has_tools else "skill"
        for old, new in (("WITHOUT any skill", f"WITH the current {what} (version {version}, shown below)"),
                         ("Without a skill the student fully solved", "With the current skill the student fully solved"),
                         ("judged that it solved", "judged that, with the current skill, it solved")):
            base = base.replace(old, new, 1)
        cur = (f"THE CURRENT {what.upper()} (version {version}) THAT THE STUDENT {'USED' if self.has_tools else 'FOLLOWED'} IN "
               f"EVERY EPISODE BELOW:\n{harness.strip()}\n\n")
        i = base.index("EPISODES:\n")
        base = base[:i] + cur + base[i:]
        if self.has_tools:
            base = (base.replace("WRITE THE SKILL OR TOOL as a markdown file", "WRITE THE REVISED SKILL OR TOOL (the next "
                                 "version) as a markdown file", 1)
                    .replace("WRITE THE SKILL WITH ITS TOOL as a markdown file", "WRITE THE REVISED SKILL WITH ITS TOOL (the "
                             "next version) as a markdown file", 1))
        else:
            base = base.replace("WRITE THE SKILL as a markdown file", "WRITE THE REVISED SKILL (the next version) as a "
                                "markdown file", 1)
        j = base.index("Rules:\n") + len("Rules:\n")
        return base[:j] + (REVISE_TOOLS if self.has_tools else REVISE) + base[j:]

    def reference_prompt(self, bench: str, workflow: str, attempts: list, references: list):
        """The test-time prompt: the workflow evidence with the memory of the most similar workflows [(name, text)]."""
        p = self.improver_prompt(bench, workflow, attempts)
        if p is None:
            return None
        i = p.index("EPISODES:\n")
        sk = "\n\n".join(f"--- skill of {name} ---\n{text.strip()}" for name, text in references)
        return p[:i] + REFERENCES.format(m=len(references), own="", skills=sk) + p[i:]
