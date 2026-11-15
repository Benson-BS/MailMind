from dataclasses import dataclass
from typing import Literal, Callable, Any, TypedDict, Optional


class Email(TypedDict):
    """One email from data/emails/emails.jsonl, as a plain dict:
    email["subject"], email["from"], email["to"]."""

    id: str  # "em-00001"
    thread_id: str  # "th-00001"
    reply_to: Optional[str]  # id of the email this one replies to, or None
    # "from" is also a key (it is a Python keyword, so it cannot be declared here).
    to: list[str]
    date: str  # ISO 8601, UTC, e.g. "2001-05-14T23:39:00Z"
    subject: str
    body: str

@dataclass
class DocWindow:
    doc_id: str
    start_line: int  # 1-based line number of the first line in `text`
    end_line: int  # 1-based line number of the last line in `text`
    total_lines: int
    text: str

@dataclass
class BM25Index:
    postings: dict[str, list[tuple[int, int]]]
    lengths: list[int]
    average_length: float
'''
docs = ["apple apple banana", "banana orange"]
postings = {
    "apple": [(0, 2)],
    "banana": [(0, 1), (1, 1)],
    "orange": [(1, 1)],
}
lengths = [3, 2]
average_length = 2.5
'''

class SearchResult(TypedDict):
    doc_id: str
    title: str
    snippet: str
    
@dataclass
class Tool_DESC:
    name: str
    description: str
    parameters: dict
    func: Callable[..., Any]


@dataclass
class AgentConfig:
    max_turns: int = 20
    
#这里返回的是agent的一轮对话的结果，其中包含多次模型和工具的调用
@dataclass
class AgentResult:
    content: str
    status: Literal["done", "max_turns", "token_budget", "malformed"]
    transcript: list[dict] #返回问答中的详尽事例
    
@dataclass
class ToolResult:
    content: str
    status: Literal["success", "not exist", "参数错误"]
    
    
