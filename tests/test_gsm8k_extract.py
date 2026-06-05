from scripts.train.gsm8k_extract import extract_gsm8k_answer


def test_gold_hash():
    assert extract_gsm8k_answer("Janet sells the eggs ... #### 18") == "18"


def test_last_number_with_commas_and_dollar():
    assert extract_gsm8k_answer("So the total is $1,200 in all.") == "1200"


def test_trailing_period():
    assert extract_gsm8k_answer("The answer is 42.") == "42"


def test_none_when_no_number():
    assert extract_gsm8k_answer("no digits here") is None
