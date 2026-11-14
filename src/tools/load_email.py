import json
from typing import List, cast

from src.config.agent_types import Email


def load_emails() -> List[Email]:
    emails: List[Email] = []

    with open("data/emails/emails.jsonl", "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                emails.append(json.loads(line))

    return emails