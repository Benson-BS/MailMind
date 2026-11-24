"""课程提供的会话运行器。

``run_persona`` 运行模拟用户与智能体的对话，并对结果评分。

1. 为用例中的每场会话构建 ``AgentConfig``，调用 ``make_session(config)``
   创建新的智能体会话。同一用例的各场会话共用 ``memory_dir``，各自使用独立的工作目录。
2. 模拟用户发出消息，通过 ``session.send(message)`` 获取智能体的回复，
   再读取回复。模拟用户认为目标已达成、选择放弃或用完发言轮次时，会话结束。
3. 运行用例配置的程序化检查，再由裁判模型评分。

评估通过需要智能体正常运行、所有检查通过，并且裁判模型判定通过。
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
import traceback
from pathlib import Path
from typing import Callable, Optional, Sequence

from src.config import sandbox

from .agent import EvaluationConfig, EvaluationUser
from .checks import run_checks
from .judge import judge_record
from .llm import LM, BudgetExceeded, OutputTruncated
from .persona import Persona
from .persona_user import PersonaUser
from .record import ConversationRecord, JudgeResult, SessionRecord

# 所有评估会话共用的智能体运行上限。
EVAL_MAX_TURNS = 20  # 每条用户消息的模型调用次数上限
EVAL_CONTEXT_BUDGET = 16_000  # 单次模型调用的上下文令牌预算

# 模拟用户模型和裁判模型的设置。模型拒绝 temperature 参数时，使用服务商的默认采样设置。
# 对话内容相同且命中缓存时，会复用先前的模型回复。令牌上限包含模型的隐藏推理令牌。
USER_MAX_TOKENS = 1000
JUDGE_MAX_TOKENS = 2500


def eval_salt() -> str:
    """读取参与评估模型缓存键计算的 ``CS329Z_EVAL_SALT``。

    设置一个新的值可生成新的缓存键，用来比较多次评估之间的结果差异。
    """
    return os.environ.get("CS329Z_EVAL_SALT", "")


class RetryingLM:
    """为模拟用户和裁判模型处理短回复被截断的情况。

    回复因达到 ``max_tokens`` 被截断时，将令牌预算加倍并重试一次。
    如果重试仍然被截断，就返回已生成的文本。
    """

    def __init__(self, role: str, *, max_tokens: int, salt: str = "", tag: str) -> None:
        self.lm = LM(role, max_tokens=max_tokens, salt=salt, tag=tag)
        self.retry = LM(role, max_tokens=2 * max_tokens, salt=salt, tag=tag)
        self.model = self.lm.model

    def __call__(self, messages) -> str:
        try:
            return self.lm(messages)
        except OutputTruncated:
            try:
                return self.retry(messages)
            except OutputTruncated as exc:
                return exc.text

    def usage(self) -> dict:
        a, b = self.lm.usage(), self.retry.usage()
        return {k: (a[k] + b[k] if isinstance(a[k], (int, float)) else a[k]) for k in a}


def make_user_lm(persona: Persona) -> RetryingLM:
    """创建扮演 ``persona`` 的模型，用例 ID 参与缓存键计算以区分各用例。"""
    return RetryingLM(
        "user",
        max_tokens=USER_MAX_TOKENS,
        salt=f"persona:{persona.id}:{eval_salt()}",
        tag=f"sim_eval/{persona.id}/user",
    )


def make_judge_lm(persona: Persona) -> RetryingLM:
    return RetryingLM(
        "judge", max_tokens=JUDGE_MAX_TOKENS, salt=eval_salt(), tag=f"sim_eval/{persona.id}/judge"
    )


def build_config(user: PersonaUser, memory_dir: Path, workspace: Path) -> EvaluationConfig:
    return EvaluationConfig(
        max_turns=EVAL_MAX_TURNS,
        context_budget=EVAL_CONTEXT_BUDGET,
        memory_dir=memory_dir,
        user=EvaluationUser(user),
        workspace=workspace,
    )


def _make_workspace(path: Path) -> Path:
    """在会话的工作目录中准备供终端读取的语料。"""
    path.mkdir(parents=True)
    if sandbox.EMAILS_PATH.is_file():
        archive = path / "emails.jsonl"
        shutil.copyfile(sandbox.EMAILS_PATH, archive)
        if os.name != "nt":
            archive.chmod(0o444)
    return sandbox.make_workspace(path)


def _usage(lms: Sequence) -> dict:
    """汇总多个语言模型对象的用量。"""
    total = {"calls": 0, "cached_calls": 0, "cost_usd": 0.0, "cost_untracked": False}
    for lm in lms:
        usage = lm.usage() if hasattr(lm, "usage") else {}
        for key in ("calls", "cached_calls", "cost_usd"):
            total[key] += usage.get(key, 0)
        total["cost_untracked"] = total["cost_untracked"] or usage.get("cost_untracked", False)
    return total


def converse(
    persona: Persona,
    make_session: Callable[[EvaluationConfig], object],
    user_lm: Callable,
    workdir: Optional[Path] = None,
    memory_out: Optional[Path] = None,
) -> ConversationRecord:
    """运行 ``persona`` 的所有会话，返回尚未评分的记录。

    智能体抛出一般异常时，结束对话并将错误保存到 ``record.error``。
    ``NotImplementedError`` 表示适配器尚未实现，``BudgetExceeded`` 表示达到费用上限。
    这两类异常会继续抛给调用方。
    """
    record = ConversationRecord(persona_id=persona.id, category=persona.category)
    user = PersonaUser(persona, user_lm)
    root = Path(tempfile.mkdtemp(prefix=f"sim-{persona.id}-", dir=workdir))
    memory_dir = root / "memory"
    memory_dir.mkdir()
    try:
        for index in range(len(persona.sessions)):
            user.start_session(index)
            session_record = SessionRecord(index=index + 1, mode=persona.mode, dialogue=user.events)
            record.sessions.append(session_record)
            session = None
            try:
                workspace = _make_workspace(root / f"workspace-{index + 1}")
                session = make_session(build_config(user, memory_dir, workspace))
                while True:
                    kind, message = user.next_turn()
                    if kind != "message":
                        session_record.ending = kind
                        break
                    result = session.send(message)
                    user.hear(result.content, result.status)
            except (NotImplementedError, BudgetExceeded):
                raise
            except Exception:
                record.error = traceback.format_exc()
                session_record.ending = "agent_error"
            finally:
                if session is not None:
                    session_record.transcript = list(getattr(session, "transcript", []) or [])
            if record.error:
                break
    finally:
        if memory_out is not None:  # 导出智能体保存的记忆，供后续检查
            shutil.rmtree(memory_out, ignore_errors=True)
            shutil.copytree(memory_dir, memory_out)
        shutil.rmtree(root, ignore_errors=True)
    return record


def score(persona: Persona, record: ConversationRecord, judge_lm: Callable) -> ConversationRecord:
    """填写记录的检查结果、裁判结果和通过状态（``checks``、``judge``、``success``）。"""
    record.checks = run_checks(record, persona.checks)
    if record.error:
        record.judge = JudgeResult(False, "Not judged: the agent raised an exception.", "")
    else:
        record.judge = judge_record(persona, record, judge_lm)
    record.success = record.error is None and not record.failed_checks and record.judge.passed
    return record


def run_persona(
    persona: Persona,
    make_session: Callable[[EvaluationConfig], object],
    *,
    user_lm: Optional[Callable] = None,
    judge_lm: Optional[Callable] = None,
    agent_lms: Sequence = (),
    workdir: Optional[Path] = None,
    memory_out: Optional[Path] = None,
) -> ConversationRecord:
    """运行 ``persona`` 与智能体的对话，并对结果评分。

    - ``make_session`` 接收 ``AgentConfig``，每场会话调用一次，返回一个提供
      ``send(message) -> AgentResult`` 方法和 ``transcript`` 属性的对象。
    - ``user_lm`` 和 ``judge_lm`` 默认使用评估配置指定的模拟用户模型和裁判模型。
    - ``agent_lms`` 是智能体在对话中使用的模型对象。传入这些对象可将其用量记录到
      ``record.usage["agent"]``。
    - ``workdir`` 指定临时记忆目录和会话工作目录的创建位置。
    - ``memory_out`` 指定导出记忆的目录。会话结束后先复制 ``memory_dir``，再清理临时目录。
    """
    user_lm = user_lm or make_user_lm(persona)
    judge_lm = judge_lm or make_judge_lm(persona)
    start = time.time()
    record = converse(persona, make_session, user_lm, workdir, memory_out)
    score(persona, record, judge_lm)
    record.seconds = round(time.time() - start, 2)
    record.usage = {"agent": _usage(agent_lms), "user": _usage([user_lm]), "judge": _usage([judge_lm])}
    return record
