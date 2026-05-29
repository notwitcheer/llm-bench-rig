from lib.agentic.sandbox import run_code_action

def test_runs_code_and_captures_result():
    code = "result = calc('2+2')"
    out = run_code_action(code)
    assert out.ok is True
    assert out.result == 4.0

def test_tool_call_flows_into_result():
    code = "result = read_file('/data/config.txt')"
    out = run_code_action(code)
    assert "vram=32GB" in out.result

def test_timeout_is_caught():
    out = run_code_action("while True: pass", timeout=2)
    assert out.ok is False
    assert out.error == "timeout"

def test_missing_result_is_error():
    out = run_code_action("x = 1")
    assert out.ok is False
    assert "no `result`" in out.error
