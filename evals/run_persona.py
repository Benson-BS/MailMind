"""运行第 01 号用例的两轮文档问答。"""

import json
from pathlib import Path

from src.agent.session import Session
from src.config.agent_types import AgentConfig
from src.tools.build_tools import build_tools
from evals.load_persona import load_personas

def main() -> None:
    personas = load_personas()
    for persona in personas:
        opening = personas[0]["sessions"][0]["opening"]

        config = AgentConfig()
        
        session = Session(config)

        for question in (opening, "when is the postmortem draft due?"):
            print(f"用户：{question}", flush=True)
            result = session.send(question)
            print(f"智能体：{result.content}\n", flush=True)
            if result.status != "done":
                raise SystemExit(f"会话结束，状态为 {result.status}")


if __name__ == "__main__":
    main()
