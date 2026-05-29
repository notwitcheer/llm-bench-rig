from lib.agentic.longcontext import build_haystack, approx_tokens

def test_needle_is_present_at_depth():
    hay = build_haystack("SECRET-123", target_tokens=2000, depth=0.5)
    assert "SECRET-123" in hay
    assert approx_tokens(hay) >= 1800

def test_depth_places_needle_in_middle_third():
    hay = build_haystack("MARKER", target_tokens=3000, depth=0.5)
    pos = hay.index("MARKER") / len(hay)
    assert 0.3 < pos < 0.7
