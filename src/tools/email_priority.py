from src.config.agent_types import Email
import re

from openai import OpenAI
system_prompt = (
    "从收件人的角度判断邮件优先级。"
    "ignore：广告、新闻简报、全员通知，或不需要收件人处理的自动通知；"
    "即使措辞紧急，也归为 ignore。"
    "urgent：明确需要该收件人处理，且期限是当天、下一个工作日，"
    "或不超过两个工作日。"
    "normal：其他写给该收件人、值得阅读的工作或私人邮件。"
    "第一行只输出 urgent、normal 或 ignore；第二行输出原因。"
)
def classify_email(email: Email, lm: OpenAI) -> dict:
    content = (
        f"日期：{email['date']}\n"
        f"发件人：{email['from']}\n"
        f"收件人：{', '.join(email['to'])}\n"
        f"主题：{email['subject']}\n"
        f"正文：{email['body']}"
    )
    res = {"category": "", "reason": ""}
    response = lm.chat.completions.create(
        model = 'deepseek-chat',
        messages = [
            {
                "role": "system",
                "content": system_prompt
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