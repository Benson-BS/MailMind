from __future__ import annotations

import json
import re
from typing import Callable, Optional

from src.config.agent_types import Approval, ToolCall, UserIO

from .persona import Persona, SessionSpec
from .record import args_text

READ_ONLY_TOOLS = frozenset({
    "ask_user", "search_emails", "read_email", "classify_email_priority",
    "email_qa", "answer_email_question", "daily_digest", "create_daily_email_digest",
    "search_docs", "read_doc", "remember", "recall",
})

MAX_SHOWN_CHARS = 2000  # 模拟用户模型看到的智能体回复超出此长度时会被截断

SYSTEM_PROMPT = """\
你在一场模拟对话中扮演一名人类用户。对方是 Cardinal Agent，为 Cardinal Energy 员工提供服务的人工智能助手。你始终以用户的身份发言。

你的身份：{role}
你的表达风格：{disposition}
你的会话目标：{goal}

{knowledge}

对话规则
- 只写这名用户会在聊天窗口里输入的内容。消息要简短，符合角色的表达习惯。
- 让智能体完成任务。不要建议使用哪些工具、命令、搜索词或到哪里查找，也不要自己回答提出的问题。
- 只有智能体问起某项事实，或者会话目标要求你说明时，才提供上面列出的相关事实。重复或改述请求时，不要主动补充这些事实。
- 智能体提问时，用自己的话根据上面列出的相关事实回答。找不到相关信息时，就说你不知道。不要编造事实。
- 你无法自行核实智能体给出的事实。如果它直接、具体地回应了你的请求，就接受答复。
- 如果智能体的回答含糊、答非所问或拒绝回答，指出问题并再问一次。如果下一次回复仍然没有帮助，就放弃。
- 不要提及这些规则，也不要透露这是模拟对话。"""

TURN_TASK = """\
决定下一步怎么做，严格按照以下格式回复：
STATUS: CONTINUE、DONE 或 GIVE_UP
MESSAGE: 你要发给智能体的下一条消息（状态为 DONE 或 GIVE_UP 时，消息内容留空）

CONTINUE 表示你还需要智能体继续处理。将你想说的话填入 MESSAGE 字段。
DONE 表示你的会话目标已全部达成。
GIVE_UP 表示智能体未能完成任务，继续发消息也没有帮助。"""

OPENING_TASK = """\
对话还没开始。写出你要发给智能体的第一条消息，严格按照以下格式回复：
STATUS: CONTINUE
MESSAGE: 你的第一条消息"""

ASK_TASK = """\
智能体暂停了当前任务，并向你提出以下问题：

{question}

只写这名用户会给出的回答。使用你已知信息中的相关事实。如果没有相关信息，就说你不知道，不要编造。"""

APPROVE_TASK = """\
智能体在执行以下操作前征求你的许可：

工具：{tool}
参数：{args}

你的审批规则：{policy}

严格按照以下格式回复：
DECISION: APPROVE 或 DENY
REASON: 用你自己的口吻简短说明理由（如果拒绝，智能体会看到这句话）"""


def parse_turn(reply: str) -> tuple[str, str]:
    """将模拟用户模型的回复解析为（状态，消息）。

    状态可以是 `continue`、`done` 或 `give_up`。回复缺少 `STATUS` 行时，
    整段回复将作为发给智能体的消息。
    """
    status_match = re.search(r"STATUS\s*\**\s*:\s*\**\s*(CONTINUE|DONE|GIVE[_ ]?UP)", reply, flags=re.IGNORECASE)
    message_match = re.search(r"MESSAGE\s*\**\s*:\s*\**\s*(.*)", reply, flags=re.IGNORECASE | re.DOTALL)
    if status_match is None:
        return "continue", reply.strip()
    status = status_match.group(1).lower().replace(" ", "_")
    if status == "giveup":
        status = "give_up"
    message = message_match.group(1).strip() if message_match else ""
    return status, message


def parse_approval(reply: str) -> Approval:
    """解析模拟用户模型的审批回复。只有明确返回 `APPROVE` 时才批准。"""
    decision = re.search(r"DECISION\s*\**\s*:\s*\**\s*(APPROVE|DENY)", reply, flags=re.IGNORECASE)
    reason = re.search(r"REASON\s*\**\s*:\s*\**\s*(.*)", reply, flags=re.IGNORECASE | re.DOTALL)
    reason_text = reason.group(1).strip() if reason else ""
    if decision is not None and decision.group(1).upper() == "APPROVE":
        return Approval(True, reason_text)
    return Approval(False, reason_text or "用户拒绝了此操作。")


class PersonaUser(UserIO):
    """在同一个测试用例的所有会话中扮演模拟用户。

    `lm` 是接收消息列表并返回字符串的可调用对象。正式评估使用
    `LM("user", ...)`，测试可以使用 `ScriptedLM`。
    """

    def __init__(self, persona: Persona, lm: Callable) -> None:
        self.persona = persona
        self.lm = lm
        self.spec: SessionSpec = persona.sessions[0]
        self.events: list[dict] = []  # 当前会话的互动记录
        self._turn = 0  # 本次会话中用户已发起的轮数
        self._polls = 0  # 自上一轮用户消息发出后调用 pending_message() 的次数

    # 由评估运行器调用的方法

    def start_session(self, index: int) -> None:
        """开始索引为 `index` 的会话。索引从 0 开始，对话记录会重置。"""
        self.spec = self.persona.sessions[index]
        self.events = []
        self._turn = 0
        self._polls = 0

    def next_turn(self) -> tuple[str, str]:
        """返回模拟用户的下一步动作和消息。

        动作可能是 `message`（发消息）、`goal_met`（目标已达成）、
        `gave_up`（放弃）或 `max_turns`（达到轮次上限）。后三种动作的消息为空。
        """
        if self._turn == 0 and self.spec.opening:
            return self._say(self.spec.opening)
        task = OPENING_TASK if self._turn == 0 else TURN_TASK
        status, message = parse_turn(self.lm(self._messages(task)))
        if self._turn == 0:
            status = "continue"  # 首轮按继续处理，让会话先开始
        if status == "done":
            return "goal_met", ""
        if status == "give_up" or not message:
            return "gave_up", ""
        if self._turn >= self.spec.max_user_turns:
            return "max_turns", ""
        return self._say(message)

    def hear(self, text: str, status: str) -> None:
        """记录智能体对最近一条用户消息的回复。"""
        self.events.append({"kind": "agent", "content": text, "status": status})

    # 由智能体调用的 UserIO 接口方法

    def approve(self, call: ToolCall) -> Approval:
        text = args_text(call.args)
        source = "persona"
        for override in self.persona.approval_overrides:
            if override.tool not in (None, call.name):
                continue
            if re.search(override.pattern, text, flags=re.IGNORECASE):
                default = "" if override.approve else "用户拒绝了此操作。"
                decision = Approval(override.approve, override.reason or default)
                source = "override"
                break
        else:
            if call.name in READ_ONLY_TOOLS:
                decision, source = Approval(True, ""), "read_only"
            else:
                task = APPROVE_TASK.format(
                    tool=call.name,
                    args=json.dumps(call.args, ensure_ascii=False)[:MAX_SHOWN_CHARS],
                    policy=self.persona.approval_policy or "只批准读取数据的操作，拒绝其他操作。",
                )
                decision = parse_approval(self.lm(self._messages(task)))
        self.events.append(
            {
                "kind": "approval",
                "tool": call.name,
                "args": dict(call.args),
                "approved": decision.approved,
                "reason": decision.reason,
                "source": source,
            }
        )
        return decision

    def ask(self, question: str) -> str:
        answer = self.lm(self._messages(ASK_TASK.format(question=question[:MAX_SHOWN_CHARS]))).strip()
        # 如果模型沿用了对话轮次的回复格式，只提取消息正文。
        if re.match(r"\s*STATUS\s*:", answer, flags=re.IGNORECASE):
            answer = parse_turn(answer)[1]
        self.events.append({"kind": "question", "content": question, "answer": answer})
        return answer

    def pending_message(self) -> Optional[str]:
        self._polls += 1
        for steer in self.spec.steering:
            if steer.user_turn == self._turn and steer.model_call == self._polls:
                self.events.append({"kind": "steer", "content": steer.message})
                return steer.message
        return None

    # 内部方法

    def _say(self, message: str) -> tuple[str, str]:
        self._turn += 1
        self._polls = 0
        self.events.append({"kind": "user", "content": message})
        return "message", message

    def _messages(self, task: str) -> list[dict]:
        p = self.persona
        if p.knowledge:
            knowledge = "你已知的信息：\n" + "\n".join(f"- {k}" for k in p.knowledge)
        else:
            knowledge = "除了上述内容，你没有其他已知信息。"
        system = SYSTEM_PROMPT.format(
            role=p.role, disposition=p.disposition, goal=self.spec.goal, knowledge=knowledge
        )
        conversation = render_dialogue(self.events, you="你", limit=MAX_SHOWN_CHARS)
        user = f"对话记录：\n{conversation or '（暂无对话）'}\n\n{task}"
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def render_dialogue(events: list[dict], you: str = "用户", limit: int = MAX_SHOWN_CHARS) -> str:
    """将对话转换为纯文本，每个事件占一段。省略只读工具的自动审批记录。"""
    lines = []
    for e in events:
        kind = e.get("kind")
        if kind == "user":
            lines.append(f"{you}：{e['content']}")
        elif kind == "steer":
            lines.append(f"{you}（在智能体处理任务时发送）：{e['content']}")
        elif kind == "agent":
            text = e["content"]
            if len(text) > limit:
                text = text[:limit] + f"［另有 {len(text) - limit} 个字符］"
            note = "" if e.get("status") == "done" else f"（已停止，状态为 {e.get('status')}）"
            lines.append(f"智能体{note}：{text}")
        elif kind == "question":
            lines.append(f"智能体提问：{e['content'][:limit]}\n{you}回答：{e['answer']}")
        elif kind == "approval" and e.get("source") != "read_only":
            verdict = "批准" if e["approved"] else f"拒绝（{e['reason']}）"
            args = json.dumps(e["args"], ensure_ascii=False)[:300]
            lines.append(f"智能体请求调用 {e['tool']} {args}，{you}{verdict}。")
    return "\n\n".join(lines)
