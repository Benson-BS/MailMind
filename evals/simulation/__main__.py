"""Run the simulated-user evaluation from the repository root."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from .agent import agent_factory
from .evaluate import DEFAULT_CONCURRENCY, PERSONA_DIR, format_table, new_run_dir, run_eval
from .persona import load_personas
from .record import ConversationRecord
from .runner import make_judge_lm, score


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="python -m evals.simulation", description=__doc__)
    parser.add_argument("personas", nargs="?", type=Path, default=PERSONA_DIR,
                        help="directory of persona JSON files")
    parser.add_argument("--slice", type=int, default=int(os.getenv("CS329Z_EVAL_SLICE") or 0),
                        help="run only the first N personas")
    parser.add_argument("--only", help="filter by persona id or category")
    parser.add_argument("--out", type=Path, help="directory for the conversation records")
    parser.add_argument("--concurrency", type=int,
                        default=int(os.getenv("CS329Z_EVAL_CONCURRENCY", DEFAULT_CONCURRENCY)),
                        help="conversations to run at the same time")
    parser.add_argument("--salt", help="change the user and judge cache keys")
    parser.add_argument("--verbose", action="store_true", help="show all judge reasons")
    parser.add_argument("--rescore", type=Path, help="score saved records again using the judge model")
    args = parser.parse_args(argv)

    if args.salt is not None:
        os.environ["CS329Z_EVAL_SALT"] = args.salt
    personas = load_personas(args.personas)
    if args.only:
        personas = [persona for persona in personas if args.only in persona.id or args.only in persona.category]
    if args.slice:
        personas = personas[:args.slice]
    if not personas:
        parser.error(f"No persona files match in {args.personas}.")

    if args.rescore is not None:
        records = []
        for persona in personas:
            path = args.rescore / f"{persona.id}.json"
            if not path.is_file():
                continue
            record = ConversationRecord.from_json(path.read_text(encoding="utf-8"))
            judge = make_judge_lm(persona)
            score(persona, record, judge)
            record.usage["judge"] = judge.usage()
            path.write_text(record.to_json(), encoding="utf-8")
            records.append(record)
        if not records:
            parser.error(f"No saved records match in {args.rescore}.")
        print(format_table(records, verbose=args.verbose))
        return 0

    factory = agent_factory()
    out_dir = args.out or new_run_dir()
    records = run_eval(personas, factory, concurrency=args.concurrency, out_dir=out_dir)
    print(format_table(records, verbose=args.verbose))
    print(f"Records saved in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
