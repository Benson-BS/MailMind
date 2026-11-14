import json
import os
import re

import pytest

from src.config.model import client
from src.tools.email_digest import daily_digest
from src.tools.load_email import load_emails


def contains(text: str, alias: str) -> bool:
    def normalize(value: str) -> str:
        value = re.sub(r"(?<=\d),(?=\d{3}\b)", "", value.lower())
        return " ".join(re.findall(r"[a-z0-9]+", value))

    needle = normalize(alias)
    return bool(needle) and f" {needle} " in f" {normalize(text)} "


@pytest.fixture(scope="module")
def live_digests():
   

    with open("tests/fixtures/digest_days.json", encoding="utf-8") as file:
        days = json.load(file)

    
    
    days = days[:3]

    by_id = {email["id"]: email for email in load_emails()}

    return [
        (
            day,
            daily_digest([by_id[email_id] for email_id in day["email_ids"]], client),
        )
        for day in days
    ]


def test_live_length(live_digests):
    for day, digest in live_digests:
        assert isinstance(digest, str), day["date"]
        assert digest.strip(), f'{day["date"]}: 摘要为空'
        assert len(digest.split()) <= 300, (
            f'{day["date"]}: 摘要有 {len(digest.split())} 个词'
        )


def test_live_mentions_every_urgent_item(live_digests):
    for day, digest in live_digests:
        for item in day["urgent"]:
            for aliases in item["must_mention"]:
                assert any(contains(digest, alias) for alias in aliases), (
                    f'{day["date"]}: 未覆盖紧急邮件 {item["id"]}；'
                    f"需要出现其中一个名称：{aliases}\n摘要：{digest}"
                )


def test_live_mentions_nothing_from_ignore_pile(live_digests):
    for day, digest in live_digests:
        for item in day["ignore"]:
            for term in item["must_not_mention"]:
                assert not contains(digest, term), (
                    f'{day["date"]}: 摘要提到了忽略邮件 '
                    f'{item["id"]} 中的 {term!r}\n摘要：{digest}'
                )