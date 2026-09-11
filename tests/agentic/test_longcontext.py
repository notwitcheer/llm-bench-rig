from lib.agentic.longcontext import build_haystack, approx_tokens, server_token_count

def test_needle_is_present_at_depth():
    hay = build_haystack("SECRET-123", target_tokens=2000, depth=0.5)
    assert "SECRET-123" in hay
    assert approx_tokens(hay) >= 1800

def test_depth_places_needle_in_middle_third():
    hay = build_haystack("MARKER", target_tokens=3000, depth=0.5)
    pos = hay.index("MARKER") / len(hay)
    assert 0.3 < pos < 0.7


class _Resp:
    def __init__(self, n): self._n = n
    def raise_for_status(self): pass
    def json(self): return {"tokens": list(range(self._n))}


class _Http:
    def __init__(self, n): self.n = n; self.calls = []
    def post(self, url, json=None, timeout=None):
        self.calls.append((url, json)); return _Resp(self.n)


class _Client:
    def __init__(self, n, url="http://127.0.0.1:8090/v1/chat/completions"):
        self.url = url; self._client = _Http(n)


def test_server_token_count_uses_tokenize_endpoint():
    c = _Client(4321)
    assert server_token_count(c, "hello") == 4321
    url, body = c._client.calls[0]
    assert url == "http://127.0.0.1:8090/tokenize" and body["content"] == "hello"


def test_server_token_count_none_without_url_or_http():
    class Bare: pass
    assert server_token_count(Bare(), "x") is None
    c = _Client(1); c._client = None
    assert server_token_count(c, "x") is None


def test_server_token_count_none_on_http_error():
    c = _Client(1)
    def boom(*a, **k): raise RuntimeError("no tokenize")
    c._client.post = boom
    assert server_token_count(c, "x") is None
