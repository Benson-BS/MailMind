"""运行第 01 号用例的两轮文档问答。"""

import json
from pathlib import Path

from src.agent.session import Session
from src.config.agent_types import AgentConfig
from src.tools.build_tools import build_tools
from evals.load_persona import load_personas
from evals.persona import PersonaUser

def main() -> None:
    personas = load_personas()
    personas = list(load_personas())[:1]
    for persona in personas:
        session_data = persona.sessions[0]
        persona_user = PersonaUser(persona)
        session = Session(AgentConfig(user=persona_user))

        question = session_data["opening"]
        max_user_turns = session_data.get("max_user_turns", 6)

        for user_turn in range(1, max_user_turns + 1):
            print(f"用户：{question}", flush=True)
            result = session.send(question)
            print(f"智能体：{result.content}\n", flush=True)

            if result.status != "done":
                raise SystemExit(f"会话结束，状态为 {result.status}")

            if user_turn == max_user_turns:
                break

            question = persona_user.next_message(result.content)
            if question is None:
                break


if __name__ == "__main__":
    main()
