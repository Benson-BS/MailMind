"""运行第 01 号用例并保存会话记录。"""

import json
from pathlib import Path
from tempfile import mkdtemp

from evals.load_persona import load_personas
from evals.persona import PersonaUser
from src.agent.session import Session
from src.config.agent_types import AgentConfig


def main() -> None:
    personas = list(load_personas())[:1]
    if not personas:
        return

    runs_dir = Path("runs")
    runs_dir.mkdir(exist_ok=True)
    output_dir = Path(mkdtemp(prefix="sim-", dir=runs_dir))

    for persona in personas:
        session_data = persona.sessions[0]
        persona_user = PersonaUser(persona)
        session = Session(AgentConfig(user=persona_user))

        question = session_data["opening"]
        max_user_turns = session_data.get("max_user_turns", 6)
        turns = []

        for user_turn in range(1, max_user_turns + 1):
            print(f"用户：{question}", flush=True)
            result = session.send(question)
            turns.append({
                "user": question,
                "agent": result.content,
                "status": result.status,
            })
            print(f"智能体：{result.content}\n", flush=True)

            if result.status != "done" or user_turn == max_user_turns:
                break

            question = persona_user.next_message(result.content)
            if question is None:
                break

        output_path = output_dir / f"{persona.id}.json"
        record = {
            "persona_id": persona.id,
            "turns": turns,
            "transcript": session.transcript,
        }
        output_path.write_text(
            json.dumps(record, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"会话记录已保存到 {output_path}", flush=True)

        if turns and turns[-1]["status"] != "done":
            raise SystemExit(f"会话结束，状态为 {turns[-1]['status']}")


if __name__ == "__main__":
    main()
