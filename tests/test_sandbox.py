from config.sandbox import run_terminal



result = run_terminal("echo hello")
print(result)  # 调试时查看完整结果

assert not result.rejected
assert result.exit_code == 0
assert result.stdout.strip() == "hello"