from config.model import LLM
from tools.register import Tool_Register

lm = LLM()
#tools = Tool_Register()
reply = lm([
        {
            "role": "system",
            "content": (
                "这是多个英文邮件的聚合结果，帮我写摘要。"
                "摘要最多300个英文单词，必须提到每一项紧急事项。"
                "每封紧急邮件中提及的公司、项目或地点名称必须明确写出。"
                "不得提及ignore类邮件中的任何内容。"
                "使用英文回答。"
            )
        },
        {
            "role": "user",
            "content": "你好，你是什么模型"
        }
    ],
    )

print(reply)