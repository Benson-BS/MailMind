from dataclasses import dataclass, field
from typing import Literal, Callable, Any, TypedDict, Optional
from pathlib import Path
from src.config.user import UserIO

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
    context_budget: int = 16_000
    workspace: Path | None = Path("./agent_workspace")
    terminal_timeout: float = 10.0
    user: UserIO | None = None
    
    
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
    
@dataclass
class TerminalResult:
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None  # None if the command was rejected or timed out
    timed_out: bool = False
    rejected: bool = False  # True if the allowlist check refused the command, or there is no shell
    reason: str = ""  # why it was rejected or killed; "" otherwise
'''
正常运行结束：`rejected=False`、`timed_out=False`，`exit_code` 是数字；即使这个数字非零，命令也确实运行过。
超时：`timed_out=True`、`exit_code=None`，`reason` 说明超时；`stdout` 和 stderr 仍可能包含超时前已经产生的内容。
被拒绝：`rejected=True`、`exit_code=None`，`reason` 说明拒绝原因；命令没有运行，所以输出通常为空。
'''


@dataclass
class Approval:
    approved: bool
    reason: str = ""
    
@dataclass
class ToolCall:
    name: str
    args: dict = field(default_factory=dict)