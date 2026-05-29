from lib.agentic.mock_tools import TOOLS, build_namespace

def test_tools_are_deterministic():
    ns = build_namespace()
    assert ns["web_search"]("rtx 5090 vram") == ns["web_search"]("rtx 5090 vram")

def test_web_search_hash_fallback_is_deterministic():
    ns = build_namespace()
    # unknown query exercises the sha256 fallback path, not the _SEARCH dict
    first = ns["web_search"]("some unknown query xyz")
    assert first == ns["web_search"]("some unknown query xyz")
    assert "ref=" in first

def test_extract_handles_class_selector():
    ns = build_namespace()
    html = "<span class='a price b'>3600</span>"
    assert ns["extract"](html, ".price") == "3600"

def test_read_file_returns_seeded_content():
    ns = build_namespace()
    assert ns["read_file"]("/data/config.txt") == "vram=32GB\nmodel=local\n"

def test_extract_pulls_value():
    ns = build_namespace()
    html = "<div id='price'>3600</div>"
    assert ns["extract"](html, "#price") == "3600"

def test_registry_lists_signatures():
    assert len(TOOLS) == 5
    assert any(t["name"] == "web_search" for t in TOOLS)
    assert all({"name", "signature", "description"} <= set(t) for t in TOOLS)
