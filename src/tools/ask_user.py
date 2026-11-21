from src.config.agent_types import Tool_DESC
from src.config.user import UserIO


def create_ask_user_tool(user: UserIO) -> Tool_DESC:
    def ask_user(question: str) -> str:
        return user.ask(question)

    return Tool_DESC(
        name="ask_user",
        description=(
            "Ask the user a clarifying question when required information "
            "is ambiguous or missing. Do not use this tool when the request "
            "can be completed from the available context."
        ),
        parameters={
            "question": {
                "type": "string",
                "description": "A clear and specific question to ask the user.",
            }
        },
        func=ask_user,
    )