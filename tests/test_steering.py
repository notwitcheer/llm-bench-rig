from lib.steering import pick_budget, rebadge_config, parse_timing, timing_summary

QWEN_CFG = {
    "architectures": ["Qwen2ForCausalLM"], "model_type": "qwen2",
    "hidden_size": 3584, "intermediate_size": 18944, "num_attention_heads": 28,
    "num_hidden_layers": 28, "num_key_value_heads": 4, "rms_norm_eps": 1e-06,
    "rope_theta": 10000.0, "vocab_size": 152064, "max_position_embeddings": 4096,
    "tie_word_embeddings": False, "torch_dtype": "bfloat16", "hidden_act": "silu",
    "bos_token_id": 151643, "eos_token_id": 151643, "initializer_range": 0.02,
}

def test_pick_budget_fits_cap():
    assert pick_budget(step_seconds=120.0, cap_minutes=40.0) == 20

def test_pick_budget_clamps():
    assert pick_budget(step_seconds=1.0) == 30      # ceil
    assert pick_budget(step_seconds=9999.0) == 5    # floor

def test_rebadge_is_llama_with_biases():
    cfg = rebadge_config(QWEN_CFG)
    assert cfg["architectures"] == ["LlamaForCausalLM"]
    assert cfg["model_type"] == "llama"
    assert cfg["attention_bias"] is True and cfg["mlp_bias"] is True
    for k in ("hidden_size", "intermediate_size", "num_hidden_layers",
              "num_key_value_heads", "rope_theta", "vocab_size", "rms_norm_eps"):
        assert cfg[k] == QWEN_CFG[k]
    assert cfg["rope_scaling"] is None

def test_rebadge_missing_key_raises():
    bad = dict(QWEN_CFG); del bad["rope_theta"]
    try:
        rebadge_config(bad); assert False
    except KeyError:
        pass

def test_parse_timing_roundtrip():
    lines = ["noise", "STEER-TIMING step=1 rollout_s=90.5 grade_s=4.2 update_s=1.1",
             "STEER-TIMING step=2 rollout_s=88.0 grade_s=4.0 update_s=1.0"]
    recs = parse_timing(lines)
    assert len(recs) == 2 and recs[0]["step"] == 1 and recs[1]["rollout_s"] == 88.0

def test_timing_summary_shares():
    recs = parse_timing(["STEER-TIMING step=1 rollout_s=90.0 grade_s=8.0 update_s=2.0"])
    s = timing_summary(recs)
    assert abs(s["rollout_share"] - 0.90) < 1e-9
    assert abs(s["update_share"] - 0.02) < 1e-9
