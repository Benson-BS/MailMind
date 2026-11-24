"""Live simulated-user evaluation for the email agent."""

import os

import pytest

from src.config.sandbox import REPO_ROOT

from evals.simulation.agent import agent_factory
from evals.simulation.evaluate import (
    DEFAULT_CONCURRENCY,
    PERSONA_DIR,
    format_table,
    new_run_dir,
    required_successes,
    run_eval,
)
from evals.simulation.persona import load_personas


@pytest.mark.live
def test_live_simulated_users(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    personas = load_personas(PERSONA_DIR)
    limit = int(os.getenv("CS329Z_EVAL_SLICE") or 0)
    if limit:
        personas = personas[:limit]
    assert personas, f"No persona files selected from {PERSONA_DIR}"

    factory = agent_factory()
    out_dir = new_run_dir()
    records = run_eval(
        personas,
        factory,
        concurrency=int(os.getenv("CS329Z_EVAL_CONCURRENCY", DEFAULT_CONCURRENCY)),
        out_dir=out_dir,
    )
    table = format_table(records)
    print(f"\n{table}\nConversation records saved in {out_dir}")

    successes = sum(record.success for record in records)
    needed = required_successes(len(records))
    assert successes >= needed, (
        f"sim_eval scored {successes}/{len(records)}, requires {needed}; logs in {out_dir}\n{table}"
    )
