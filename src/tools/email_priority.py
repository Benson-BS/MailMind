from src.config.agent_types import Email
import re
from src.config.model import client
from openai import OpenAI
def classify_email(email: Email, lm: OpenAI) -> dict:
    content = email['body']
    res = {"category": "", "reason": ""}
    response = client.chat.completions.create(
        model = 'deepseek-chat',
        messages = [
            {
                "role": "system",
                "content": (
                    "根据邮件内容分类。第一行只输出 urgent、normal、ignore、unknown "
                    "之一，第二行输出原因。"
                ),
            },
            {"role": "user", "content": content},
        ],  
    )
    reply = response.choices[0].message.content 
    match = re.search(r"\b(urgent|normal|ignore)\b", reply)
    if match:
        category = match.group()
    else:
        category = "unknown"
    lines = reply.splitlines()

    # 第二行及之后的内容作为原因
    reason = "\n".join(lines[1:]).strip()

    # 防止模型只输出类别，或返回空字符串
    if not reason:
        reason = "模型没有提供原因"
    res["category"] = category
    res["reason"] = reason
    
    return res