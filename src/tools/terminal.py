from config import sandbox
from src.config.agent_types import AgentConfig, Tool_DESC
from pathlib import Path

class termial_tool():
    def __init__(self, config: AgentConfig):
        self.config = config

    def _limit_observation(
        self,
        text: str,
        workspace: str,
        max_chars: int = 20_000,
    ) -> str:
        """
        大输出策略：
        1. 小输出直接返回；
        2. 大输出保留头尾；
        3. 尽可能把完整输出保存到 workspace，方便模型之后 grep/read。
        """
        if len(text) <= max_chars:
            return text


        notice = (
            "\n\n[terminal output truncated]\n"
            f"Original output contained {len(text)} characters.\n"
            f"Full output was saved to {workspace}.\n"
            "Use grep, sed, head, or tail through run_terminal to inspect it.\n"
        )
        output_dir = workspace / ".terminal_outputs.txt"
        with open(output_dir, "w", encoding='uft-8') as f:
            f.write(text)
        # 为提示文字和换行预留空间。
        remaining = max_chars - len(notice)
        head_size = max(0, remaining // 2)
        tail_size = max(0, remaining - head_size)

        return (
            text[:head_size]
            + notice
            + "\n[tail of output]\n"
            + text[-tail_size:]
        )
        
    def __call__(self, cmd: str) -> str:
        result = sandbox.run_terminal(
                cmd,
                workspace=self.config.workspace,
                timeout=self.config.terminal_timeout,
            )
        
        exit_code = "None" if result.exit_code is None else str(result.exit_code)
        parts = [
            f"stdout:\n{result.stdout}",
            f"stderr:\n{result.stderr}",
            f"exit_code: {exit_code}",
        ]

        
        if result.reason:
            parts.append(f"reason: {result.reason}")
            
        str_result = "\n\n".join(parts)
        if not result.rejected and not result.timed_out:
            show = self._limit_observation(
                str_result,
                workspace=self.config.workspace,
                max_chars = 20_000
            )
        else:
            show = result
            
        return show