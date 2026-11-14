import json
import os

import pytest

from src.config.model import client
from src.tools.email_priority import classify_email

def test_priority_live():
    with open("tests/fixtures/priority_gold.json", encoding="utf-8") as file:
        gold = json.load(file)

        gold = gold[:10]

    wrong = []
    unknown = []

    for item in gold:
        result = classify_email(item["email"], client)

        assert result["category"] in {"urgent", "normal", "ignore", "unknown"}
        assert isinstance(result["reason"], str) and result["reason"].strip()

        if result["category"] == "unknown":
            unknown.append(item["email"]["id"])
        if result["category"] != item["category"]:
            wrong.append(
                f'{item["email"]["id"]}: '
                f'期望 {item["category"]}，实际 {result["category"]}'
            )

    assert not unknown, f"返回 unknown 的邮件：{unknown}"
    assert (len(gold) - len(wrong)) / len(gold) >= 0.85, "\n".join(wrong)