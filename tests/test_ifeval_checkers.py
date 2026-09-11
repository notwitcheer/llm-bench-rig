"""IFEval native checkers + the eval loop with a fake client (no network).

One or more asserts per instruction family; the eval-loop tests patch the
`datasets` module so nothing is downloaded.
"""
import json
import sys
import types

import pytest

from lib.evals import ifeval
from lib.evals.ifeval import (CHECKERS, IFEvalEval, check_instruction,
                              count_sentences, count_words, score_response)


# --- length_constraints ---

def test_number_words():
    assert check_instruction("length_constraints:number_words", "one two three four",
                             {"num_words": 3, "relation": "at least"}) is True
    assert check_instruction("length_constraints:number_words", "one two three four",
                             {"num_words": 3, "relation": "less than"}) is False
    assert count_words("hello, world! it's 3am") == 5


def test_number_sentences():
    text = "First one. Second one! Third one? Fourth"
    assert count_sentences(text) == 4
    assert check_instruction("length_constraints:number_sentences", text,
                             {"num_sentences": 5, "relation": "less than"}) is True
    assert check_instruction("length_constraints:number_sentences", text,
                             {"num_sentences": 5, "relation": "at least"}) is False


def test_number_paragraphs_with_divider():
    text = "para one\n***\npara two\n***\npara three"
    assert check_instruction("length_constraints:number_paragraphs", text,
                             {"num_paragraphs": 3}) is True
    assert check_instruction("length_constraints:number_paragraphs", text,
                             {"num_paragraphs": 2}) is False
    # blank middle paragraph is a failure
    assert check_instruction("length_constraints:number_paragraphs",
                             "a\n***\n\n***\nb", {"num_paragraphs": 3}) is False


def test_nth_paragraph_first_word():
    text = "Alpha starts here.\n\nBravo is second.\n\nCharlie closes."
    kw = {"num_paragraphs": 3, "nth_paragraph": 2, "first_word": "bravo"}
    assert check_instruction("length_constraints:nth_paragraph_first_word", text, kw) is True
    assert check_instruction("length_constraints:nth_paragraph_first_word", text,
                             {**kw, "first_word": "alpha"}) is False
    assert check_instruction("length_constraints:nth_paragraph_first_word", text,
                             {**kw, "num_paragraphs": 2}) is False


# --- detectable_format ---

def test_json_format():
    assert check_instruction("detectable_format:json_format", '{"a": 1}', {}) is True
    assert check_instruction("detectable_format:json_format",
                             '```json\n{"a": [1, 2]}\n```', {}) is True
    assert check_instruction("detectable_format:json_format", "not json", {}) is False


def test_number_bullet_lists():
    text = "intro\n* one\n* two\n- three"
    assert check_instruction("detectable_format:number_bullet_lists", text,
                             {"num_bullets": 3}) is True
    assert check_instruction("detectable_format:number_bullet_lists", text,
                             {"num_bullets": 2}) is False


def test_constrained_response():
    assert check_instruction("detectable_format:constrained_response",
                             "My answer is maybe.", {}) is True
    assert check_instruction("detectable_format:constrained_response",
                             "Probably yes", {}) is False


def test_number_highlighted_sections():
    text = "some *highlight one* and **bold two** and * * empty"
    assert check_instruction("detectable_format:number_highlighted_sections", text,
                             {"num_highlights": 2}) is True
    assert check_instruction("detectable_format:number_highlighted_sections", text,
                             {"num_highlights": 3}) is False


def test_multiple_sections():
    text = "SECTION 1\nfoo\nSECTION 2\nbar"
    assert check_instruction("detectable_format:multiple_sections", text,
                             {"section_spliter": "SECTION", "num_sections": 2}) is True
    assert check_instruction("detectable_format:multiple_sections", text,
                             {"section_spliter": "SECTION", "num_sections": 3}) is False


def test_title():
    assert check_instruction("detectable_format:title", "<<My Title>>\nbody", {}) is True
    assert check_instruction("detectable_format:title", "<< >>\nbody", {}) is False
    assert check_instruction("detectable_format:title", "no title", {}) is False


# --- detectable_content ---

def test_postscript():
    assert check_instruction("detectable_content:postscript", "body\n\nP.S. call me",
                             {"postscript_marker": "P.S."}) is True
    assert check_instruction("detectable_content:postscript", "body\n\nP.P.S. again",
                             {"postscript_marker": "P.P.S"}) is True
    assert check_instruction("detectable_content:postscript", "body only",
                             {"postscript_marker": "P.S."}) is False


def test_number_placeholders():
    text = "Dear [name], welcome to [city]."
    assert check_instruction("detectable_content:number_placeholders", text,
                             {"num_placeholders": 2}) is True
    assert check_instruction("detectable_content:number_placeholders", text,
                             {"num_placeholders": 3}) is False


# --- keywords ---

def test_keywords_existence():
    assert check_instruction("keywords:existence", "The cat sat on the Mat.",
                             {"keywords": ["cat", "mat"]}) is True
    assert check_instruction("keywords:existence", "The cat sat.",
                             {"keywords": ["cat", "dog"]}) is False


def test_keywords_frequency():
    text = "go go go stop"
    assert check_instruction("keywords:frequency", text,
                             {"keyword": "go", "frequency": 3, "relation": "at least"}) is True
    assert check_instruction("keywords:frequency", text,
                             {"keyword": "go", "frequency": 3, "relation": "less than"}) is False


def test_forbidden_words():
    assert check_instruction("keywords:forbidden_words", "clean text here",
                             {"forbidden_words": ["dirty"]}) is True
    assert check_instruction("keywords:forbidden_words", "some Dirty text",
                             {"forbidden_words": ["dirty"]}) is False


def test_letter_frequency():
    text = "banana"
    assert check_instruction("keywords:letter_frequency", text,
                             {"letter": "a", "let_frequency": 3, "let_relation": "at least"}) is True
    assert check_instruction("keywords:letter_frequency", text,
                             {"letter": "a", "let_frequency": 3, "let_relation": "less than"}) is False


# --- language ---

def test_response_language_unsupported_without_langdetect(monkeypatch):
    monkeypatch.setitem(sys.modules, "langdetect", None)  # force ImportError
    assert check_instruction("language:response_language", "bonjour", {"language": "fr"}) is None


def test_response_language_with_fake_langdetect(monkeypatch):
    fake = types.ModuleType("langdetect")
    fake.detect = lambda text: "fr"
    monkeypatch.setitem(sys.modules, "langdetect", fake)
    assert check_instruction("language:response_language", "bonjour", {"language": "fr"}) is True
    assert check_instruction("language:response_language", "bonjour", {"language": "de"}) is False


# --- change_case ---

def test_english_capital_and_lowercase():
    assert check_instruction("change_case:english_capital", "ALL CAPS HERE.", {}) is True
    assert check_instruction("change_case:english_capital", "Not all caps", {}) is False
    assert check_instruction("change_case:english_lowercase", "all lower here.", {}) is True
    assert check_instruction("change_case:english_lowercase", "Not lower", {}) is False


def test_capital_word_frequency():
    text = "this is VERY IMPORTANT stuff"
    assert check_instruction("change_case:capital_word_frequency", text,
                             {"capital_frequency": 2, "capital_relation": "at least"}) is True
    assert check_instruction("change_case:capital_word_frequency", text,
                             {"capital_frequency": 2, "capital_relation": "less than"}) is False


# --- startend ---

def test_end_checker():
    assert check_instruction("startend:end_checker", "blah blah. Is there anything else?",
                             {"end_phrase": "Is there anything else?"}) is True
    assert check_instruction("startend:end_checker", "blah blah. Bye.",
                             {"end_phrase": "Is there anything else?"}) is False


def test_quotation():
    assert check_instruction("startend:quotation", '"wrapped response"', {}) is True
    assert check_instruction("startend:quotation", '"half wrapped', {}) is False


# --- punctuation ---

def test_no_comma():
    assert check_instruction("punctuation:no_comma", "no commas here", {}) is True
    assert check_instruction("punctuation:no_comma", "one, comma", {}) is False


# --- combination ---

def test_repeat_prompt():
    assert check_instruction("combination:repeat_prompt", "Write a poem about X\n\nRoses...",
                             {"prompt_to_repeat": "Write a poem about X"}) is True
    assert check_instruction("combination:repeat_prompt", "Sure! Write a poem about X",
                             {"prompt_to_repeat": "Write a poem about X"}) is False


def test_two_responses():
    assert check_instruction("combination:two_responses", "first\n******\nsecond", {}) is True
    assert check_instruction("combination:two_responses", "same\n******\nsame", {}) is False
    assert check_instruction("combination:two_responses", "a\n******\nb\n******\nc", {}) is False


# --- dispatch / scoring ---

def test_all_25_dataset_instruction_ids_have_checkers():
    expected = {
        "change_case:capital_word_frequency", "change_case:english_capital",
        "change_case:english_lowercase", "combination:repeat_prompt",
        "combination:two_responses", "detectable_content:number_placeholders",
        "detectable_content:postscript", "detectable_format:constrained_response",
        "detectable_format:json_format", "detectable_format:multiple_sections",
        "detectable_format:number_bullet_lists",
        "detectable_format:number_highlighted_sections", "detectable_format:title",
        "keywords:existence", "keywords:forbidden_words", "keywords:frequency",
        "keywords:letter_frequency", "language:response_language",
        "length_constraints:nth_paragraph_first_word",
        "length_constraints:number_paragraphs", "length_constraints:number_sentences",
        "length_constraints:number_words", "punctuation:no_comma",
        "startend:end_checker", "startend:quotation",
    }
    assert set(CHECKERS) == expected
    assert check_instruction("made_up:thing", "x", {}) is None


def test_null_kwargs_are_dropped():
    # the HF release carries every kwarg key with None for unused ones
    kw = {"num_words": 2, "relation": None, "keywords": None, "letter": None}
    assert check_instruction("length_constraints:number_words", "a b c", kw) is True


def test_score_response_strict_requires_all_checked():
    ids = ["punctuation:no_comma", "detectable_format:title", "language:response_language"]
    kws = [{}, {}, {"language": "en"}]
    s = score_response("<<T>> fine text", ids, kws)
    assert s["strict"] is True and s["inst_pass"] == 2 and s["inst_checked"] == 2
    assert s["unsupported"] == 1 and s["per_instruction"] == [True, True, None]
    s = score_response("<<T>> has, comma", ids, kws)
    assert s["strict"] is False and s["inst_pass"] == 1
    # all-unsupported prompt is never a pass
    s = score_response("x", ["language:response_language"], [{"language": "en"}])
    assert s["strict"] is False and s["inst_checked"] == 0


# --- eval loop with fake client + fake datasets ---

class _Client:
    def __init__(self, answers, tokens):
        self.answers, self.tokens, self.calls = list(answers), list(tokens), []
        self.last_completion_tokens = None
        self.reasoning_fallback_count = 0

    def chat(self, messages, **kw):
        self.calls.append((messages, kw))
        i = len(self.calls) - 1
        self.last_completion_tokens = self.tokens[i]
        return self.answers[i]


_ITEMS = [
    {"key": 1, "prompt": "no commas please", "instruction_id_list": ["punctuation:no_comma"],
     "kwargs": [{"num_words": None}]},
    {"key": 2, "prompt": "title it", "instruction_id_list": ["detectable_format:title"],
     "kwargs": [{}]},
    {"key": 3, "prompt": "in french", "instruction_id_list": ["language:response_language"],
     "kwargs": [{"language": "fr"}]},
]


def _patch_datasets(monkeypatch):
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *a, **k: list(_ITEMS)
    monkeypatch.setitem(sys.modules, "datasets", fake)
    monkeypatch.setitem(sys.modules, "langdetect", None)


def test_eval_loop_scores_and_writes_sidecars(tmp_path, monkeypatch):
    _patch_datasets(monkeypatch)
    client = _Client(["fine no comma", "no title here", "bonjour"], [10, 1024, 5])
    res = IFEvalEval(client, results_dir=tmp_path).evaluate()
    assert res["metric"] == "prompt_strict_acc"
    assert res["total"] == 3 and res["correct"] == 1
    assert res["score"] == pytest.approx(33.33, abs=0.01)
    assert res["inst_checked"] == 2 and res["inst_pass"] == 1
    assert res["inst_strict_acc"] == 50.0
    assert res["unsupported_instructions"] == 1
    assert res["capped_count"] == 1
    # standing token metrics via token_summary (2026-09-11: were missing, detail.json read null)
    assert res["tokens_recorded"] == 3 and res["capped_rate"] == pytest.approx(0.3333, abs=1e-4)
    assert res["completion_tokens_total"] == 10 + 1024 + 5 and res["tokens_per_correct"] == 1039.0
    assert res["parse_failures"] == 0
    assert client.calls[0][1]["max_tokens"] == ifeval.DEFAULT_MAX_TOKENS == 1024
    assert client.calls[0][0] == [{"role": "user", "content": "no commas please"}]
    prog = json.loads((tmp_path / "ifeval_progress.json").read_text())["completed"]
    assert set(prog) == {"0", "1", "2"}
    assert prog["1"]["completion_tokens"] == 1024 and prog["1"]["capped"] is True
    assert json.loads((tmp_path / "ifeval_detail.json").read_text())["score"] == res["score"]
    assert len((tmp_path / "ifeval_tokens.jsonl").read_text().splitlines()) == 3


def test_eval_loop_resumes_from_progress(tmp_path, monkeypatch):
    _patch_datasets(monkeypatch)
    c1 = _Client(["fine no comma", "<<T>> ok", "bonjour"], [1, 2, 3])
    IFEvalEval(c1, results_dir=tmp_path).evaluate()
    c2 = _Client([], [])
    res = IFEvalEval(c2, results_dir=tmp_path).evaluate()
    assert c2.calls == []
    assert res["correct"] == 2 and res["total"] == 3


def test_limit_caps_items(monkeypatch):
    _patch_datasets(monkeypatch)
    client = _Client(["x"], [1])
    res = IFEvalEval(client, limit=1).evaluate()
    assert res["total"] == 1 and len(client.calls) == 1
