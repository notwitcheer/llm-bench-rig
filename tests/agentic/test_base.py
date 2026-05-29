from lib.agentic.base import extract_code, load_system_prompt, build_tool_doc

def test_extract_code_from_fence():
    txt = "Sure:\n```python\nresult = 1\n```\nDone."
    assert extract_code(txt) == "result = 1"

def test_extract_code_strips_think():
    txt = "<think>plan</think>\n```python\nresult = 2\n```"
    assert extract_code(txt) == "result = 2"

def test_extract_code_bare_fallback():
    assert extract_code("result = calc('1+1')") == "result = calc('1+1')"

def test_build_tool_doc_lists_signatures():
    doc = build_tool_doc()
    assert "web_search(query: str) -> str" in doc

def test_load_system_prompt_falls_back(tmp_path):
    p = tmp_path / "missing.txt"
    assert load_system_prompt(str(p)).startswith("You are")  # synthetic fallback
