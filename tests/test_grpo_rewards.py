from scripts.train.grpo_rewards import correctness_reward, format_reward


def _conv(text):
    return [{"role": "assistant", "content": text}]


def test_correctness_rewards_exact_match():
    comps = [_conv("reasoning\n#### 18"), _conv("nope\n#### 7")]
    assert correctness_reward(completions=comps, answer=["18", "18"]) == [2.0, 0.0]


def test_correctness_zero_when_gold_none():
    comps = [_conv("#### 18")]
    assert correctness_reward(completions=comps, answer=[None]) == [0.0]


def test_correctness_accepts_plain_strings():
    assert correctness_reward(completions=["#### 42"], answer=["42"]) == [2.0]


def test_format_rewards_hash_marker():
    comps = [_conv("blah #### 18"), _conv("no marker here")]
    assert format_reward(completions=comps) == [0.5, 0.0]
