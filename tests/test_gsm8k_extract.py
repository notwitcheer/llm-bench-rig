from scripts.train.gsm8k_extract import extract_gsm8k_answer


def test_gold_hash():
    assert extract_gsm8k_answer("Janet sells the eggs ... #### 18") == "18"


def test_last_number_with_commas_and_dollar():
    assert extract_gsm8k_answer("So the total is $1,200 in all.") == "1200"


def test_trailing_period():
    assert extract_gsm8k_answer("The answer is 42.") == "42"


def test_none_when_no_number():
    assert extract_gsm8k_answer("no digits here") is None


def test_strips_calc_annotations_and_prefers_answer_marker():
    # MetaMathQA-style output: <<calc>> annotations + "The answer is: X", then rambling.
    txt = ("Janet makes 9 * $2 = $<<9*2=18>>18 every day.\nThe answer is: 18\n\n"
           "Problem: something else\nThe answer is: 99")
    assert extract_gsm8k_answer(txt) == "18"  # first committed answer, not the rambled 99


def test_first_hash_answer_not_last():
    # model echoes the prompt's #### format then rambles into a new problem
    assert extract_gsm8k_answer("reasoning #### 18 then a new problem #### 7") == "18"
