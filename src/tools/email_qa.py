"""Multi-hop question answering over retrieved emails."""

import json
import re
from typing import Callable, Literal, Annotated, List

from src.config.agent_types import Email

from openai import OpenAI

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from src.tools.email_search import EmailSearchEngine

system_prompt = """帮我完成任务，这个任务是要从搜索引擎的结果中根据搜索的问题找到答案，
                   如果搜索引擎的结果中包含答案，进行总结，并严格按照以下json格式回答：
                   只有检索到明确写出答案的邮件时，才返回 Finish；
                    support 只能填写直接支持答案的邮件 ID。
                    如果当前结果不足以回答，返回 Search。
                    新的检索词使用结果中的人名、项目名或地点，加上要查的事项；
                    不要重复上一轮的完整问题，也不要用 Finish 表示“未找到”。
                   
                   {"action":"Finish", "summery":"这里填写答案","support":"这里返回一个列表list[str]每一个元素为支持结果的邮件id"}
                   如果没有找到答案，严格按照以下格式输出
                    {"action":"Search", "question":"填写你认为给搜索引擎的提示词"}"""

class Answer(BaseModel):
    model_config = ConfigDict(extra='forbid')
    action: Literal["Finish"]
    summery: str
    support: List[str]
    
class Search(BaseModel):
    model_config = ConfigDict(extra='forbid')
    action: Literal["Search"]
    question: str
    

def answer_email_question(
    question: str, search_engine: EmailSearchEngine, lm: OpenAI
) -> dict:
    query = question
    
    for _ in range(3):
        search_result = search_engine.search(query, 3)
        user_prompt = f"目前搜索的问题是{query}, 使用搜索工具搜索的结果是{search_result}\
            帮我进行寻找答案，并找出相关的证据"
        response = lm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        reply = response.choices[0].message.content or ""
        AgentResponse = Annotated[
            Answer | Search,
            Field(discriminator="action")
        ]

        
        try:
            response = TypeAdapter(AgentResponse).validate_json(reply)
        except ValueError:
            return {"answer": None, "support": []}
        if response.action == "Finish":
            return {
                "answer": response.summery,
                "support": response.support
            }

        elif response.action == "Search":
            query = response.question
            
    return {"answer": None, "support": []}    

    
