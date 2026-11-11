from typing import List
from src.config.tool_types import Tool_DESC
class Tool_Register():
    def __init__(self, list_tool_desc: List[Tool_DESC] | None = None):
        self.tool_list: List[dict] = []
        for tool_desc in list_tool_desc or []:
            self.tool_list.append(self._to_tool_protocol(tool_desc))
        
    def _to_tool_protocol(self, tool_desc: Tool_DESC) -> dict:
        
        
        
        return {
            "type": "function",
            "function": {
                "name": tool_desc.name,
                "description": tool_desc.description,
                "parameters": {
                    "type": "object",
                    "properties": tool_desc.parameters,
                    "required": ["location"]
                }
            }
        }
