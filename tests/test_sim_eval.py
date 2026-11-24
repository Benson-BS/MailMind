"""Offline regression test for approval and recovery in an agent session."""

import json
import os
from types import SimpleNamespace
from unittest.mock import Mock

from dotenv import dotenv_values

from src.config.agent_types import AgentConfig, Approval, Tool_DESC
from src.config.user import ScriptedUser


def _tool_reply(content: str, name: str, args: dict) -> SimpleNamespace:
    return SimpleNamespace(
        content=content,
        tool_calls=[
            SimpleNamespace(
                function=SimpleNamespace(name=name, arguments=json.dumps(args))
            )
        ],
    )


def test_scripted_persona_conversation(monkeypatch, tmp_path):
    # The SDK client is created on import and requires a key even with mocked replies.
    if not os.getenv("deepseek_api_key") and not dotenv_values().get("deepseek_api_key"):
        monkeypatch.setenv("deepseek_api_key", "offline-test")
    from src.agent.session import Session
    from src.config.model import client

    denial_reason = "Do not save notes. Just tell me."
    note_args = {"text": "helpdesk question"}
    lookup_args = {"key": "it_helpdesk"}
    extension = "extension 4410"
    answer = f"The IT helpdesk is at {extension}."

    write_note = Mock(return_value="Saved note.")
    lookup = Mock(return_value=extension)
    tools = [
        Tool_DESC(
            name="write_note",
            description="Save a note.",
            parameters={"text": {"type": "string"}},
            func=write_note,
            read_only=False,
        ),
        Tool_DESC(
            name="lookup",
            description="Look up an office directory entry.",
            parameters={"key": {"type": "string"}},
            func=lookup,
            read_only=True,
        ),
    ]
    replies = [
        _tool_reply("Saving a note first.", "write_note", note_args),
        _tool_reply("Looking it up.", "lookup", lookup_args),
        SimpleNamespace(content=answer, tool_calls=None),
    ]
    responses = [SimpleNamespace(choices=[SimpleNamespace(message=reply)]) for reply in replies]
    complete = Mock(side_effect=responses)
    monkeypatch.setattr(client.chat.completions, "create", complete)

    user = ScriptedUser(approvals=[Approval(approved=False, reason=denial_reason)])
    session = Session(AgentConfig(user=user, workspace=tmp_path), tools)
    result = session.send("What is the IT helpdesk extension?")

    assert user.log == [("approve", "write_note", note_args, False)]
    write_note.assert_not_called()
    lookup.assert_called_once_with(**lookup_args)

    tool_events = [
        event
        for event in result.transcript
        if event.get("type") in {"tool_call", "tool_result"}
    ]
    assert tool_events == [
        {"type": "tool_call", "name": "write_note", "args": note_args},
        {
            "type": "tool_result",
            "name": "write_note",
            "status": "denied",
            "content": denial_reason,
        },
        {"type": "tool_call", "name": "lookup", "args": lookup_args},
        {
            "type": "tool_result",
            "name": "lookup",
            "status": "success",
            "content": extension,
        },
    ]

    assert complete.call_count == 3
    after_denial = complete.call_args_list[1].kwargs["messages"]
    after_lookup = complete.call_args_list[2].kwargs["messages"]
    assert any(message["content"] == denial_reason for message in after_denial)
    assert any(message["content"] == extension for message in after_lookup)
    assert result.status == "done"
    assert result.content == answer
