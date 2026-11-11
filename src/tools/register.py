from typing import List
from config.agent_types import Tool_DESC, ToolResult
from jsonschema import validate
from jsonschema import ValidationError
class Tool_Register():
    def __init__(self, list_tool_desc: List[Tool_DESC] | None = None):
        self.tool_list: List[dict] = []
        self.tool_map = {}
        
        for tool_desc in list_tool_desc or []:
            self.tool_list.append(self._to_tool_protocol(tool_desc))
            self.tool_map[tool_desc.name] = tool_desc
        
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

    def execute(self, tool_name: str, tool_parameters: dict) -> ToolResult:
        #找配置
        tool = self.tool_map.get(tool_name) 
        if tool is None:
            return ToolResult(content=' ', status='not exist')
        
        #验证
        schema = self._to_tool_protocol(tool)["function"]["parameters"]
        
        try:
            validate(instance=tool_parameters, schema=schema)
        except ValidationError as exc:
            return ToolResult(content=exc.message, status="参数错误")
        
        #执行
        tool_result = tool.func(**tool_parameters)
        return ToolResult(content = tool_result, status='success')