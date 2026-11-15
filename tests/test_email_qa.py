import json
import re
from pathlib import Path

from src.config.model import client
from src.tools.email_search import EmailSearchEngine
from src.tools.email_qa import answer_email_question
from src.tools.load_email import load_emails
from src.tools.email_search import make_email_search

def contains_answer(text: str, aliases: list[str]) -> bool:
    def normalize(value: str) -> str:
        value = re.sub(r"(?<=\d),(?=\d{3}\b)", "", value.lower())
        return " ".join(re.findall(r"[a-z0-9]+", value))

    normalized = f" {normalize(text)} "
    return any(f" {normalize(alias)} " in normalized for alias in aliases)


def test_email_qa_live():
    engine = EmailSearchEngine(load_emails())

    fixture = Path(__file__).parent / "fixtures" / "email_qa.json"
    gold = json.loads(fixture.read_text(encoding="utf-8"))[:10]

    correct = 0
    wrong = []
    missing_support = []

    for item in gold:
        calls = 0
        returned_ids: set[str] = set()
        result = answer_email_question(item["quenstion"], engine.search, client)

        assert calls <= 3, f'{item["id"]}: 搜索了 {calls} 次'
        assert isinstance(result, dict)
        assert result.get("answer") is None or isinstance(result["answer"], str)
        assert isinstance(result.get("support"), list), f'{item["id"]}: {result}'
        assert set(result["support"]) <= returned_ids, (
            f'{item["id"]}: 引用了未搜索到的邮件'
        )

        answer = result["answer"]
        if isinstance(answer, str) and contains_answer(answer, item["answers"]):
            correct += 1
            if not set(result["support"]) & set(item["support"]):
                missing_support.append(f'{item["id"]}: {result}')
        else:
            wrong.append(f'{item["id"]}: 期望 {item["answers"]}，实际 {answer!r}')

    assert correct / len(gold) >= 0.70, "\n".join(wrong)
    assert not missing_support, "\n".join(missing_support)