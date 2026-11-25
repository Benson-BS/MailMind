from evals.load_persona import Persona
from src.config.model import LLM
import json
from src.config.agent_types import UserIO, ToolCall, Approval
from typing import override

class PersonaUser(UserIO):
    def __init__(self, persona: Persona):
        self.id = persona.id
        self.category = persona.category
        self.role = persona.role
        self.disposition = persona.disposition
        
        self.sessions = persona.sessions
        self.knowledge = persona.knowledge
        self.approval = persona.approval
        self.deny_rules = persona.deny_rules
        self.llm = LLM()
        knowledge = "\n".join(f"- {fact}" for fact in self.knowledge) or "暂无额外信息"
        self.system_prompt = (
            "你在扮演一位正在使用邮件助手的真实用户。"
            "请用第一人称说话，沿用用户开场时使用的语言。\n\n"
            f"你的身份\n{self.role}\n\n"
            f"你的说话风格\n{self.disposition}\n\n"
            f"你知道的信息\n{knowledge}\n\n"
            "根据当前目标和对话内容，按需要补充你知道的信息。"
            "遇到自己不知道的细节，就直接说不知道。"
            "只输出你要发给助手的消息正文。"
        )
        session_data = self.sessions[0]
        self.goal = session_data["goal"]
        self.chat_logs = [
            {"speaker": "user", "content": session_data["opening"]},
        ]
        
    def next_message(self, agent_reply: str):
        
        self.chat_logs.append({"speaker": "agent", "content": agent_reply})

        prompt = (
            f"{self.system_prompt}\n\n"
            f"本场会话的目标\n{self.goal}\n\n"
            "根据对话记录决定下一条用户消息。"
            "还有问题没回答时，追问缺失的信息；"
            "还有下一项任务时，继续提出对应的问题。"
            "目标完成后只输出 [END]。"
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(self.chat_logs, ensure_ascii=False)},
        ]
        reply = self.llm(messages).strip()
        
        if reply == "[END]":
            return None
        
        self.chat_logs.append({"speaker": "user", "content": reply})
        
        return reply
        
    @override
    def ask(self, question: str) -> str:
        
        self.chat_logs.append({"speaker": "agent", "content": question})
        prompt = (
            f"{self.system_prompt}\n\n"
            f"本场会话的目标\n{self.goal}\n\n"
            "对话的最后一条是助手刚提出的澄清问题。"
            "只回答这个问题，提供相关的已知信息。"
            "如果你不知道答案，就用自己的口吻说不知道。"
        )
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(self.chat_logs, ensure_ascii=False)},
        ]
        reply = self.llm(messages).strip()
        self.chat_logs.append({"speaker": "user", "content": reply})
        return reply
    
    @override    
    def approve(self, call: ToolCall) -> Approval:
        
        
        
        
        
        
    @override
    def pending_message(self) -> Optional[str]: