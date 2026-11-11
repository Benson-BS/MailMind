from dataclasses import dataclass
from typing import Literal, Callable, Any
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