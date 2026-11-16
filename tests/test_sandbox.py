from src.config.sand_box import run_terminal


def test_run_terminal_echo():
    result = run_terminal("echo hello")
    print(result)  # 调试时查看完整结果

    assert not result.rejected
    assert result.exit_code == 0
    assert result.stdout.strip() == "hello"