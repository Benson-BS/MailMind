from typing import List
from src.config.agent_types import Tool_DESC, ToolResult, ToolCall, UserIO
from jsonschema import validate
from jsonschema import ValidationError
import json
class Tool_Register():
    def __init__(self, list_tool_desc: List[Tool_DESC] | None = None,
                 user: UserIO | None = None):
        self.tool_list: List[dict] = []
        self.tool_map = {}
        self.user = user
        
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
                }
            }
        }

    
    def get(self, tool_name: str) -> Tool_DESC | None:
        return self.tool_map.get(tool_name)
    
    def _request_approval(self, tool: Tool_DESC, tool_parameters: dict) -> ToolResult | None:
        if tool.read_only or self.user is None:
            return None

        approval = self.user.approve(
            ToolCall(
                name=tool.name,
                args=tool_parameters,
            )
        )


        if approval.approved:
            return None

        return ToolResult(
            content=(
                approval.reason
                or "The user declined this action."
            ),
            status="denied",
        )
    
    
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
        #如果是可以写的工具，需要问user是否允许执行
        denial = self._request_approval(tool, tool_parameters)
        if denial is not None:
            return denial
        #返回的结果是TollResult的格式
        #执行
        tool_result = tool.func(**tool_parameters)
        if not isinstance(tool_result, str):
            tool_result = json.dumps(tool_result, ensure_ascii=False, default=str)
        return ToolResult(content = tool_result, status='success')