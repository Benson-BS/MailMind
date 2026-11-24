"""Run many personas and report the results (course-provided)."""

from __future__ import annotations

import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional, Sequence

from .persona import Persona
from .record import ConversationRecord
from .runner import run_persona

EVAL_SIZE = 20  # conversations in the full evaluation
FULL_CREDIT = 16  # successful conversations needed for full credit
MAX_POINTS = 10
DEFAULT_CONCURRENCY = 4
PERSONA_DIR = Path(__file__).resolve().parents[1] / "personas"
RUNS_DIR = PERSONA_DIR.parents[1] / "runs"

# An agent factory is called once per persona and returns
# (make_session, agent_lms): the function that creates a session from an
# AgentConfig, and the LM objects that session uses (for the cost column).
AgentFactory = Callable[[Persona], tuple[Callable, Sequence]]


def run_eval(
    personas: Sequence[Persona],
    agent_factory: AgentFactory,
    *,
    concurrency: int = DEFAULT_CONCURRENCY,
    out_dir: Optional[Path] = None,
    workdir: Optional[Path] = None,
) -> list[ConversationRecord]:
    """Run every persona (``concurrency`` conversations at a time) and
    return the records in the order of ``personas``. If ``out_dir`` is
    given, each record is saved there as ``<persona id>.json``, the persona's
    memory directory as ``<persona id>.memory/``, and the table as
    ``summary.txt``."""

    def one(persona: Persona) -> ConversationRecord:
        make_session, agent_lms = agent_factory(persona)
        memory_out = out_dir / f"{persona.id}.memory" if out_dir is not None else None
        record = run_persona(persona, make_session, agent_lms=agent_lms, workdir=workdir, memory_out=memory_out)
        if out_dir is not None:
            (out_dir / f"{persona.id}.json").write_text(record.to_json(), encoding="utf-8")
        return record

    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        records = list(pool.map(one, personas))
    if out_dir is not None:
        (out_dir / "summary.txt").write_text(format_table(records) + "\n", encoding="utf-8")
    return records


def required_successes(n: int, full_credit: int = FULL_CREDIT, eval_size: int = EVAL_SIZE) -> int:
    """Successes needed for full credit when ``n`` conversations are run:
    the same fraction as 16 of 20, rounded down. A single-case run must pass."""
    return max(1, (full_credit * n) // eval_size) if n else 0


def points(successes: int, n: int = EVAL_SIZE) -> int:
    """The handout's rule for (sim_eval): full credit (10 points) at 16 or
    more successes out of 20, one point less for each conversation below
    16, never below zero. For ``n`` other than 20 the full-credit line is
    ``required_successes(n)``."""
    return max(0, MAX_POINTS - max(0, required_successes(n) - successes))


def format_table(records: Sequence[ConversationRecord], verbose: bool = False) -> str:
    """The per-persona table and totals. Failed conversations are followed
    by the failed checks and the judge's reason; ``verbose`` prints the
    judge's reason for passed conversations too."""
    width = max([len(r.persona_id) for r in records] + [7])
    lines = [f"{'persona':<{width}}  {'category':<15}{'result':<7}{'turns':>5}{'tools':>6}{'est. USD':>9}{'sec':>7}"]
    for r in records:
        tools = sum(1 for s in r.sessions for e in s.transcript if e.get("type") == "tool_call")
        cost = "n/a" if r.usage.get("agent", {}).get("cost_untracked") else f"{r.cost_usd:.4f}"
        lines.append(
            f"{r.persona_id:<{width}}  {r.category:<15}{'PASS' if r.success else 'FAIL':<7}"
            f"{r.user_turns:>5}{tools:>6}{cost:>9}{r.seconds:>7.1f}"
        )
        if r.error:
            lines.append("    agent raised: " + r.error.strip().splitlines()[-1])
        for check in r.failed_checks:
            lines.append(f"    check failed: {check.type}: {check.detail}")
        if r.judge is not None and not r.error and (verbose or not r.judge.passed):
            lines.append(f"    judge {'PASS' if r.judge.passed else 'FAIL'}: {r.judge.reason}")
    n = len(records)
    successes = sum(r.success for r in records)
    cost = sum(r.cost_usd for r in records)
    cost_untracked = any(r.usage.get("agent", {}).get("cost_untracked") for r in records)
    lines.append("")
    lines.append(
        f"Successful conversations: {successes} of {n} "
        f"(full credit at this size needs {required_successes(n)}). "
        + (f"Estimated evaluator cost: ${cost:.4f}; direct agent API cost is unavailable."
           if cost_untracked else f"Estimated cost: ${cost:.4f}.")
    )
    if cost_untracked:
        lines.append("n/a means the direct agent model's cost was not returned by the API; check DeepSeek billing for the total.")
    else:
        lines.append("Cached model calls add no cost to the estimate.")
    if n == EVAL_SIZE:
        lines.append(f"Points under the handout's rule: {points(successes)} of {MAX_POINTS}.")
    return "\n".join(lines)


def new_run_dir(base: Path = RUNS_DIR) -> Path:
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=time.strftime("sim-%Y%m%d-%H%M%S-"), dir=base))
