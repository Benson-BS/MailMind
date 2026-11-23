"""Talk to the email agent from a terminal.

Run with::

    python -m src.config.chat
    python -m src.config.chat --out runs
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from src.agent.session import Session
from src.config.agent_types import AgentConfig
from src.config.user import ConsoleUser


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the email agent.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("runs"),
        help="directory in which to save the conversation",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("agent_workspace"),
        help="workspace exposed to the terminal tool",
    )
    args = parser.parse_args()

    config = AgentConfig(
        user=ConsoleUser(),
        workspace=args.workspace,
    )
    session = Session(config)

    print("Email Agent. Type exit or quit to leave.")
    started = time.strftime("%Y%m%d-%H%M%S")
    turns: list[dict] = []

    while True:
        try:
            message = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if message.lower() in ("exit", "quit"):
            break
        if not message:
            continue

        result = session.send(message)
        note = "" if result.status == "done" else f" [stopped: {result.status}]"
        print(f"\nagent{note}> {result.content}")
        turns.append(
            {
                "user": message,
                "agent": result.content,
                "status": result.status,
            }
        )

    if not turns:
        return

    args.out.mkdir(parents=True, exist_ok=True)
    output_path = args.out / f"chat-{started}.json"
    record = {
        "started": started,
        "turns": turns,
        "transcript": session.transcript,
    }
    output_path.write_text(
        json.dumps(record, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"Saved this conversation to {output_path}")


if __name__ == "__main__":
    main()
