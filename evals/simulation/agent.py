"""Connect the evaluation to the project's native-tool agent session."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.config import sandbox
from src.config.agent_types import AgentConfig, Approval, ToolCall, UserIO

from .agent_client import EvaluationClient
from .llm import DEEPSEEK_URL
from .persona_user import PersonaUser


@dataclass
class EvaluationConfig(AgentConfig):
    memory_dir: Path | None = None


class EvaluationUser(UserIO):
    def __init__(self, user: PersonaUser):
        self.user = user
        self.deny_rules = tuple(re.compile(rule) for rule in user.persona.deny_rules)

    def approve(self, call: ToolCall) -> Approval:
        if call.name == "run_terminal":
            cmd = call.args.get("cmd")
            if not isinstance(cmd, str) or not cmd.strip():
                return Approval(False, "A terminal call requires a non-empty command.")
            if any(rule.search(cmd) for rule in self.deny_rules):
                return Approval(False, "This command matches an evaluation deny rule.")
            if sandbox.jail() == "none":
                return Approval(False, "The evaluation requires an OS sandbox for terminal commands.")
        if self.user.persona.mode == "auto":
            return Approval(True)
        return self.user.approve(call)

    def ask(self, question: str) -> str:
        return self.user.ask(question)

    def pending_message(self) -> str | None:
        return self.user.pending_message()


class DirectAgentUsage:
    def usage(self) -> dict:
        return {"cost_untracked": True}


def agent_factory():
    """Build the default tools once and bind the workspace per session."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(sandbox.REPO_ROOT / ".env")
    key = os.getenv("deepseek_api_key") or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("Set deepseek_api_key or DEEPSEEK_API_KEY in the project .env.")
    if not sandbox.EMAILS_PATH.is_file() or not any(sandbox.DOCS_DIR.glob("*.md")):
        raise FileNotFoundError("Evaluation requires data/emails/emails.jsonl and data/docs/*.md.")

    # The original project reads the lowercase key when creating its client.
    with patch.dict(os.environ, {"deepseek_api_key": key}):
        from src.agent.session import Session
        from src.tools.build_tools import build_tools
    from src.tools.email_digest import daily_digest as summarize_emails
    from src.tools.email_priority import classify_email
    from src.tools.email_qa import answer_email_question
    from src.tools.load_email import load_emails
    from src.tools.terminal import termial_tool

    client = OpenAI(api_key=key, base_url=DEEPSEEK_URL)
    auxiliary_client = EvaluationClient(client)
    tools = build_tools(AgentConfig())
    tools_by_name = {tool.name: tool for tool in tools}
    read_email = tools_by_name["read_email"].func
    search_emails = SimpleNamespace(search=tools_by_name["search_emails"].func)

    @cache
    def email_ids_by_date() -> dict[str, list[str]]:
        dates: dict[str, list[str]] = {}
        for email in load_emails():
            dates.setdefault(email["date"][:10], []).append(email["id"])
        return dates

    def daily_digest(date: str) -> str:
        email_ids = email_ids_by_date().get(date, [])
        return summarize_emails([read_email(email_id) for email_id in email_ids], auxiliary_client)

    def classify_email_by_id(email_id: str) -> dict:
        return classify_email(read_email(email_id), auxiliary_client)

    def ask_about_emails(question: str) -> dict:
        return answer_email_question(question, search_emails, auxiliary_client)

    tools_by_name["create_daily_email_digest"] = replace(
        tools_by_name["create_daily_email_digest"],
        name="daily_digest",
        description="Create an English digest for a UTC date in YYYY-MM-DD format.",
        parameters={"date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}},
        func=daily_digest,
    )
    tools_by_name["classify_email_priority"] = replace(
        tools_by_name["classify_email_priority"], func=classify_email_by_id,
    )
    tools_by_name["answer_email_question"] = replace(
        tools_by_name["answer_email_question"], func=ask_about_emails,
    )

    def make_session(config: EvaluationConfig):
        session_tools = [
            replace(tool, func=termial_tool(config), read_only=False)
            if tool.name == "run_terminal" else tool
            for tool in tools_by_name.values()
        ]
        session = Session(config, session_tools)
        session.lm.lm = EvaluationClient(client)
        return session

    def factory(_persona):
        return make_session, [DirectAgentUsage()]

    return factory
