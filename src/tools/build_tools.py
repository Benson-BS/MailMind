"""Factory for the default tools shared by agent sessions."""

from __future__ import annotations

import json

from src.config.agent_types import AgentConfig, Tool_DESC
from src.config.model import client
from src.tools.doc_engine import DocEngine
from src.tools.email_digest import daily_digest
from src.tools.email_priority import classify_email
from src.tools.email_qa import answer_email_question
from src.tools.email_read import EmailReader
from src.tools.email_search import EmailSearchEngine
from src.tools.load_email import load_emails
from src.tools.terminal import termial_tool


def build_tools(config: AgentConfig) -> list[Tool_DESC]:
    """Build the default tools available to an agent session."""
    terminal = termial_tool(config)
    emails = load_emails()
    email_reader = EmailReader(emails)
    email_search = EmailSearchEngine(emails)
    doc_engine = DocEngine()

    def as_json(value: object) -> str:
        return json.dumps(value, ensure_ascii=False, default=str)

    def search_emails(query: str, k: int = 5) -> str:
        return as_json(email_search.search(query, k))

    def read_email(email_id: str) -> str:
        return as_json(email_reader.read(email_id))

    def classify_email_by_id(email_id: str) -> str:
        return as_json(classify_email(email_reader.read(email_id), client))

    def create_daily_digest(email_ids: list[str]) -> str:
        selected = [email_reader.read(email_id) for email_id in email_ids]
        return daily_digest(selected, client)

    def ask_about_emails(question: str) -> str:
        return as_json(answer_email_question(question, email_search, client))

    def search_docs(query: str, k: int = 5) -> str:
        return as_json(doc_engine.docs_search(query, k))

    def read_doc(doc_id: str, start_line: int = 1) -> str:
        return as_json(doc_engine.doc_read(doc_id, start_line))

    return [
        Tool_DESC(
            name="run_terminal",
            description=(
                "Run a shell command inside the sandbox. Use it to inspect "
                "workspace files or calculate answers that must be exact."
            ),
            parameters={
                "cmd": {
                    "type": "string",
                    "description": "Shell command to execute in the sandbox.",
                }
            },
            func=terminal,
            read_only=False,
        ),
        Tool_DESC(
            name="search_emails",
            description=(
                "Search the email archive by keywords. Returns matching emails "
                "with their IDs, metadata, subjects, and bodies."
            ),
            parameters={
                "query": {"type": "string", "description": "Keywords to search for."},
                "k": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5,
                },
            },
            func=search_emails,
            read_only=True,
        ),
        Tool_DESC(
            name="read_email",
            description="Read one complete email from the archive by its email ID.",
            parameters={
                "email_id": {
                    "type": "string",
                    "description": "Email ID, for example em-00001.",
                }
            },
            func=read_email,
            read_only=True,
        ),
        Tool_DESC(
            name="classify_email_priority",
            description="Classify an email as urgent, normal, or ignore and explain why.",
            parameters={
                "email_id": {
                    "type": "string",
                    "description": "ID of the email to classify.",
                }
            },
            func=classify_email_by_id,
            read_only=True,
        ),
        Tool_DESC(
            name="create_daily_email_digest",
            description="Create an English daily digest for a specified set of email IDs.",
            parameters={
                "email_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Email IDs to include in the daily digest.",
                }
            },
            func=create_daily_digest,
            read_only=True,
        ),
        Tool_DESC(
            name="answer_email_question",
            description="Answer a question that may require searching multiple related emails.",
            parameters={
                "question": {
                    "type": "string",
                    "description": "Question to answer from the email archive.",
                }
            },
            func=ask_about_emails,
            read_only=True,
        ),
        Tool_DESC(
            name="search_docs",
            description=(
                "Search the document collection and return document IDs, titles, "
                "and matching snippets."
            ),
            parameters={
                "query": {"type": "string", "description": "Keywords to search for."},
                "k": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5,
                },
            },
            func=search_docs,
            read_only=True,
        ),
        Tool_DESC(
            name="read_doc",
            description=(
                "Read a document from a 1-based start line. The result includes "
                "the returned line range and total line count."
            ),
            parameters={
                "doc_id": {"type": "string", "description": "Document ID to read."},
                "start_line": {
                    "type": "integer",
                    "description": "First line to return, using 1-based numbering.",
                    "default": 1,
                },
            },
            func=read_doc,
            read_only=True,
        ),
    ]
