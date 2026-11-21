import json
import re
from pathlib import Path

import pytest

from src.agent.session import Session
from src.config.agent_types import AgentConfig, Tool_DESC
from src.tools.terminal import termial_tool


FIXTURE = Path(__file__).parent / "fixtures" / "terminal_questions.json"
LIVE_MAX_TURNS = 15
MIN_CORRECT_FRACTION = 0.8


def normalize(value: str) -> str:
    value = value.lower()
    value = re.sub(r"(?<=\d),(?=\d{3}\b)", "", value)
    return " ".join(re.findall(r"[a-z0-9]+", value))


def contains_any(text: str, answers: list[str]) -> bool:
    normalized_text = f" {normalize(text)} "

    return any(
        f" {normalize(answer)} " in normalized_text
        for answer in answers
    )


def make_live_session() -> Session:
    config = AgentConfig(
        max_turns=LIVE_MAX_TURNS,
        terminal_timeout=30.0,
    )

    terminal = termial_tool(config)

    terminal_desc = Tool_DESC(
        name="run_terminal",
        description=(
            "Run a shell command inside the sandbox. "
            "The workspace contains emails.jsonl and a docs directory. "
            "Use terminal commands or Python to calculate exact answers. "
            "Do not guess answers that can be computed."
        ),
        parameters={
            "cmd": {
                "type": "string",
                "description": "Shell command to execute in the sandbox.",
            }
        },
        func=terminal,
    )

    return Session(config, [terminal_desc])


def terminal_commands(session: Session) -> list[str]:
    commands = []

    for event in session.transcript:
        if (
            event.get("type") == "tool_call"
            and event.get("name") == "run_terminal"
        ):
            args = event.get("args", {})

            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            commands.append(args.get("cmd", ""))

    return commands

def test_live_terminal_questions():
    assert FIXTURE.exists(), f"测试数据不存在：{FIXTURE}"

    items = json.loads(FIXTURE.read_text(encoding="utf-8"))

    rows = []

    for item in items:
        # 每个主问题使用一个全新的 Session。
        session = make_live_session()

        turns = [
            (
                item["id"],
                item["question"],
                item["answers"],
            )
        ]

        if item.get("followup"):
            followup = item["followup"]
            turns.append(
                (
                    f'{item["id"]}-followup',
                    followup["question"],
                    followup["answers"],
                )
            )

        # 主问题和它的 follow-up 使用同一个 Session。
        for turn_id, question, answers in turns:
            result = session.send(question)
            commands = terminal_commands(session)

            rows.append(
                {
                    "id": turn_id,
                    "question": question,
                    "answers": answers,
                    "answer": result.content,
                    "status": result.status,
                    "commands": commands.copy(),
                    "correct": contains_any(
                        result.content,
                        answers,
                    ),
                }
            )

    correct = sum(row["correct"] for row in rows)
    total = len(rows)
    fraction = correct / total

    lines = [
        "",
        f"terminal live: {correct}/{total} correct "
        f"({fraction:.1%}, need {MIN_CORRECT_FRACTION:.0%})",
    ]

    for row in rows:
        mark = "OK" if row["correct"] else "MISS"
        lines.append(
            f'{mark} {row["id"]}: '
            f'status={row["status"]}, '
            f'commands={len(row["commands"])}'
        )

        if not row["correct"]:
            lines.append(f'  question: {row["question"]}')
            lines.append(f'  expected: {row["answers"]}')
            lines.append(f'  actual: {row["answer"]!r}')
            lines.append(f'  commands: {row["commands"]}')

    report = "\n".join(lines)
    print(report)

    assert all(row["status"] == "done" for row in rows), report
    assert all(row["commands"] for row in rows), (
        "至少有一个问题没有调用 run_terminal。\n" + report
    )
    assert fraction >= MIN_CORRECT_FRACTION, report