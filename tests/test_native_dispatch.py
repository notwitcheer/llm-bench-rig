from lib.agentic.native.dispatch import dispatch_tool


def test_calc_dispatch():
    out = dispatch_tool("calc", {"expr": "2+40"})
    assert out["ok"] is True
    assert "42" in str(out["result"])


def test_web_search_dispatch():
    out = dispatch_tool("web_search", {"query": "rtx 5090 vram"})
    assert "32GB" in str(out["result"])


def test_execute_python_via_sandbox():
    out = dispatch_tool("execute_python", {"code": "result = calc('6*7')"})
    assert out["ok"] is True and "42" in str(out["result"])


def test_unknown_tool():
    out = dispatch_tool("nope", {})
    assert out["ok"] is False


def test_dispatch_ext_tool_read_doc():
    out = dispatch_tool("read_doc", {"doc_id": "logs-32k-early"})
    assert out["ok"] is True and "activation code" in str(out["result"])


def test_dispatch_ext_recovery_tool():
    out = dispatch_tool("read_record", {"rec_id": "007"})
    assert out["ok"] is True and "active" in str(out["result"])
