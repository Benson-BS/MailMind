"""Adapt project model requests inside an evaluation run."""

from collections import deque
from types import SimpleNamespace

from openai import OpenAI

from .llm import NON_THINKING, resolve_model


class EvaluationClient:
    def __init__(self, client: OpenAI):
        self.client = client
        self.model = resolve_model("grading")
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.complete))
        self.history: list[dict] = []
        self.messages_seen = 0
        self.pending_tool_ids: deque[str] = deque()

    def complete(self, **request):
        request["model"] = self.model
        request["extra_body"] = NON_THINKING
        if not request.get("tools"):
            return self.client.chat.completions.create(**request)

        # Session records tool results as user messages. Recover their IDs
        # from the model's previous reply before sending the next request.
        self._append_session_messages(request["messages"])
        request["messages"] = self.history.copy()
        response = self.client.chat.completions.create(**request)
        reply = response.choices[0].message
        assistant_message = {"role": "assistant", "content": reply.content or ""}
        if reply.tool_calls:
            assistant_message["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in reply.tool_calls
            ]
            self.pending_tool_ids.extend(call.id for call in reply.tool_calls)
        self.history.append(assistant_message)
        return response

    def _append_session_messages(self, messages: list[dict]) -> None:
        for message in messages[self.messages_seen:]:
            if message["role"] == "assistant":
                continue  # Already captured from the provider reply.
            if message["role"] == "user" and self.pending_tool_ids:
                self.history.append({
                    "role": "tool",
                    "tool_call_id": self.pending_tool_ids.popleft(),
                    "content": message["content"],
                })
            else:
                self.history.append(message)
        self.messages_seen = len(messages)
