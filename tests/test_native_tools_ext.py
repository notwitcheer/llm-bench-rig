from lib.agentic.native.tools_ext import vault_code, make_doc, read_doc, _DOCS


def test_vault_code_deterministic_8hex():
    c = vault_code(101)
    assert c == vault_code(101)
    assert len(c) == 8 and c.isupper()


def test_make_doc_is_deterministic_and_holds_needle():
    a = make_doc(101, 4000, 0.5)
    b = make_doc(101, 4000, 0.5)
    assert a == b                                   # pure / reproducible
    assert f"vault 101 is {vault_code(101)}" in a   # unique needle present


def test_make_doc_hits_target_token_size_within_10pct():
    # token proxy = words * 1.3; target 8000 tokens -> ~6154 words
    doc = make_doc(7, 8000, 0.25)
    approx_tokens = len(doc.split()) * 1.3
    assert 7200 <= approx_tokens <= 8800


def test_needle_depth_places_needle():
    early = make_doc(9, 4000, 0.1)
    late = make_doc(9, 4000, 0.9)
    needle = f"vault 9 is {vault_code(9)}"
    assert early.index(needle) < late.index(needle)


def test_read_doc_returns_doc_for_known_id_and_errors_unknown():
    assert "activation code" in read_doc(next(iter(_DOCS)))
    assert read_doc("nope").startswith("ERROR")
