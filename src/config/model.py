from openai import OpenAI
import os
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()

client = OpenAI(
    api_key=os.getenv('deepseek_api_key'),
    base_url="https://api.deepseek.com",
    # Brotli 1.0.9 in this environment cannot decode httpx2's Brotli responses.
    
)


class LLM():
    def __init__(self):
        self.lm = client
    
    def __call__(self, messages: List[Dict], tools=None) -> str:
        response = self.lm.chat.completions.create(
            model = "deepseek-chat",
            messages = messages,
            tools = tools
        )
        
        return response.choices[0].message
 
lm = LLM()

if __name__ == "__main__":
    
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
    ])

    print(reply)
