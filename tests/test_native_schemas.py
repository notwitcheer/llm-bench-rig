from lib.agentic.native.schemas import to_openai_tools
from lib.agentic.mock_tools import TOOLS


def test_builds_one_function_per_tool():
    out = to_openai_tools(TOOLS)
    assert len(out) == len(TOOLS)
    assert all(t["type"] == "function" for t in out)


def test_calc_tool_has_expr_param():
    out = {t["function"]["name"]: t for t in to_openai_tools(TOOLS)}
    calc = out["calc"]["function"]
    assert calc["description"]
    assert "expr" in calc["parameters"]["properties"]
    assert calc["parameters"]["properties"]["expr"]["type"] == "string"
    assert "expr" in calc["parameters"]["required"]
