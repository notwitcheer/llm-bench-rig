from scripts.embed_bench.models import MODELS, ARMS


def test_three_arms_in_fixed_order():
    assert ARMS == ["e5-small", "qwen3-text", "qwen3-vl"]


def test_every_arm_has_required_fields():
    for arm in ARMS:
        cfg = MODELS[arm]
        assert isinstance(cfg["hf_id"], str) and cfg["hf_id"]
        assert "query_prefix" in cfg          # may be "" but must be present
        assert "doc_prefix" in cfg
        assert isinstance(cfg["native_dim"], int) and cfg["native_dim"] > 0


def test_e5_uses_e5_prefixes():
    # e5 family REQUIRES the "query:"/"passage:" prefixes or quality collapses
    assert MODELS["e5-small"]["query_prefix"] == "query: "
    assert MODELS["e5-small"]["doc_prefix"] == "passage: "
