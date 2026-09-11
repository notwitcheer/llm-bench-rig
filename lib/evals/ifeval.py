"""IFEval — instruction-following evaluation via chat completions.

Zhou et al. 2023, "Instruction-Following Evaluation for Large Language Models".
541 prompts, each carrying 1-3 verifiable instructions (25 instruction types in
9 families). The model answers the prompt; each instruction is checked
programmatically against the response. Strict mode only (no loose variants
with markdown stripped / first-and-last line dropped).

Dataset: google/IFEval on HuggingFace, split `train`, fields (verified
2026-09-10 via the datasets-server rows API):
  key                 int   prompt id (1000..)
  prompt              str   the instruction prompt sent to the model
  instruction_id_list list[str]  e.g. "punctuation:no_comma"
  kwargs              list[dict] one dict per instruction; every dict carries
                      ALL 24 kwarg keys with None for the unused ones
                      (capital_frequency, capital_relation, end_phrase,
                      first_word, forbidden_words, frequency, keyword, keywords,
                      language, let_frequency, let_relation, letter,
                      nth_paragraph, num_bullets, num_highlights,
                      num_paragraphs, num_placeholders, num_sections,
                      num_sentences, num_words, postscript_marker,
                      prompt_to_repeat, relation, section_spliter)

Checkers are implemented natively below (no external `ifeval`/`nltk`/
`langdetect` dependency). `language:response_language` needs a language
detector; when `langdetect` is not importable the instruction is recorded as
`unsupported` and the prompt is scored on its remaining instructions, with the
unsupported count reported in the detail json so the number is never silently
inflated.

Metrics: `score` = prompt-level strict accuracy (every instruction in the
prompt passes), `inst_strict_acc` = instruction-level strict accuracy.
Second-tier task: reported as its own row, never folded into q_avg.
"""

import json
import re
import time
from pathlib import Path

from .base import chat_or_error, drop_errored, LLMClient, ProgressFile

DEFAULT_MAX_TOKENS = 1024
from .gpqa import CAP_SLACK, token_summary  # shared cap slack + standing token metrics
DATASET = "google/IFEval"

# --- relation helper (shared by every "at least / less than" instruction) ---

_LESS = "less than"
_AT_LEAST = "at least"


def _rel(count: int, target: int, relation: str | None) -> bool:
    relation = relation or _AT_LEAST
    if relation == _LESS:
        return count < target
    return count >= target


# --- text splitting helpers (mirrors the reference implementation) ---

def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def count_sentences(text: str) -> int:
    """Sentence count without nltk: split on ., !, ? followed by whitespace/end.

    Abbreviation-safe enough for the strict check (the reference uses the nltk
    punkt tokenizer; the two agree on ordinary prose)."""
    text = text.strip()
    if not text:
        return 0
    parts = re.split(r"(?<=[.!?])[\"')\]]*\s+", text)
    return len([p for p in parts if p.strip()])


def split_paragraphs(text: str) -> list[str]:
    """Paragraphs are separated by the markdown divider `***` (reference rule)."""
    return [p for p in re.split(r"\s?\*\*\*\s?", text) if p.strip()]


# --- instruction checkers: (response, kwargs) -> bool ---

def _len_number_words(r, kw):
    return _rel(count_words(r), kw["num_words"], kw.get("relation"))


def _len_number_sentences(r, kw):
    return _rel(count_sentences(r), kw["num_sentences"], kw.get("relation"))


def _len_number_paragraphs(r, kw):
    paragraphs = re.split(r"\s?\*\*\*\s?", r)
    n = len(paragraphs)
    for i, p in enumerate(paragraphs):
        if not p.strip():
            # a blank first/last chunk is tolerated; a blank middle one is a failure
            if i in (0, len(paragraphs) - 1):
                n -= 1
            else:
                return False
    return n == kw["num_paragraphs"]


def _len_nth_paragraph_first_word(r, kw):
    paragraphs = [p for p in re.split(r"\n\n", r) if p.strip()]
    n = len(paragraphs)
    nth = kw["nth_paragraph"]
    if n != kw["num_paragraphs"] or nth < 1 or nth > n:
        return False
    first = paragraphs[nth - 1].strip().split()
    if not first:
        return False
    word = first[0].strip().lstrip("\"'([").rstrip("\"'.,!?;:)]")
    return word.lower() == kw["first_word"].lower()


def _fmt_json(r, kw):
    text = r.strip()
    text = re.sub(r"^```(?:json|JSON)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        json.loads(text)
        return True
    except (ValueError, TypeError):
        return False


def _fmt_number_bullet_lists(r, kw):
    bullets = re.findall(r"^\s*\*[^\*].*$", r, flags=re.MULTILINE)
    bullets += re.findall(r"^\s*-.*$", r, flags=re.MULTILINE)
    return len(bullets) == kw["num_bullets"]


_CONSTRAINED = ("My answer is yes.", "My answer is no.", "My answer is maybe.")


def _fmt_constrained_response(r, kw):
    return any(opt in r for opt in _CONSTRAINED)


def _fmt_number_highlighted_sections(r, kw):
    n = 0
    for h in re.findall(r"\*[^\n\*]*\*", r):
        if h.strip("*").strip():
            n += 1
    for h in re.findall(r"\*\*[^\n\*]*\*\*", r):
        if h.strip("**").strip():
            n += 1
    return n >= kw["num_highlights"]


def _fmt_multiple_sections(r, kw):
    spliter = kw["section_spliter"]
    sections = re.split(r"\s?" + re.escape(spliter) + r"\s?\d+\s?", r)
    return len(sections) - 1 >= kw["num_sections"]


def _fmt_title(r, kw):
    titles = re.findall(r"<<[^\n]+>>", r)
    return any(t.strip("<>").strip() for t in titles)


def _content_postscript(r, kw):
    marker = kw["postscript_marker"]
    if marker == "P.P.S":
        pat = r"\s*p\.\s?p\.\s?s.*$"
    elif marker == "P.S.":
        pat = r"\s*p\.\s?s\..*$"
    else:
        pat = r"\s*" + re.escape(marker.lower()) + r".*$"
    return re.search(pat, r.lower(), flags=re.MULTILINE) is not None


def _content_number_placeholders(r, kw):
    return len(re.findall(r"\[.*?\]", r)) >= kw["num_placeholders"]


def _kw_existence(r, kw):
    low = r.lower()
    return all(re.search(r"\b" + re.escape(k.lower()) + r"\b", low) for k in kw["keywords"])


def _kw_frequency(r, kw):
    n = len(re.findall(re.escape(kw["keyword"].lower()), r.lower()))
    return _rel(n, kw["frequency"], kw.get("relation"))


def _kw_forbidden_words(r, kw):
    low = r.lower()
    return not any(re.search(r"\b" + re.escape(w.lower()) + r"\b", low)
                   for w in kw["forbidden_words"])


def _kw_letter_frequency(r, kw):
    n = r.lower().count(kw["letter"].lower())
    return _rel(n, kw["let_frequency"], kw.get("let_relation"))


def _lang_response_language(r, kw):
    try:
        import langdetect  # optional; never a rig dependency
    except ImportError:
        raise UnsupportedInstruction("langdetect not installed")
    try:
        return langdetect.detect(r) == kw["language"]
    except Exception:  # langdetect raises on empty/ambiguous text
        return False


def _case_english_capital(r, kw):
    return r.isupper()


def _case_english_lowercase(r, kw):
    return r.islower()


def _case_capital_word_frequency(r, kw):
    words = re.findall(r"\w+", r)
    n = sum(1 for w in words if w.isupper())
    return _rel(n, kw["capital_frequency"], kw.get("capital_relation"))


def _startend_end_checker(r, kw):
    return r.strip().strip('"').lower().endswith(kw["end_phrase"].strip().lower())


def _startend_quotation(r, kw):
    s = r.strip()
    return len(s) > 1 and s[0] == '"' and s[-1] == '"'


def _punct_no_comma(r, kw):
    return "," not in r


def _combo_repeat_prompt(r, kw):
    return r.strip().lower().startswith(kw["prompt_to_repeat"].strip().lower())


def _combo_two_responses(r, kw):
    parts = r.split("******")
    responses = [p.strip() for p in parts if p.strip()]
    # exactly one divider, two non-empty distinct halves
    return len(parts) == 2 and len(responses) == 2 and responses[0] != responses[1]


class UnsupportedInstruction(Exception):
    """Raised by a checker whose runtime dependency is missing; scored as unsupported."""


CHECKERS = {
    "length_constraints:number_words": _len_number_words,
    "length_constraints:number_sentences": _len_number_sentences,
    "length_constraints:number_paragraphs": _len_number_paragraphs,
    "length_constraints:nth_paragraph_first_word": _len_nth_paragraph_first_word,
    "detectable_format:json_format": _fmt_json,
    "detectable_format:number_bullet_lists": _fmt_number_bullet_lists,
    "detectable_format:constrained_response": _fmt_constrained_response,
    "detectable_format:number_highlighted_sections": _fmt_number_highlighted_sections,
    "detectable_format:multiple_sections": _fmt_multiple_sections,
    "detectable_format:title": _fmt_title,
    "detectable_content:postscript": _content_postscript,
    "detectable_content:number_placeholders": _content_number_placeholders,
    "keywords:existence": _kw_existence,
    "keywords:frequency": _kw_frequency,
    "keywords:forbidden_words": _kw_forbidden_words,
    "keywords:letter_frequency": _kw_letter_frequency,
    "language:response_language": _lang_response_language,
    "change_case:english_capital": _case_english_capital,
    "change_case:english_lowercase": _case_english_lowercase,
    "change_case:capital_word_frequency": _case_capital_word_frequency,
    "startend:end_checker": _startend_end_checker,
    "startend:quotation": _startend_quotation,
    "punctuation:no_comma": _punct_no_comma,
    "combination:repeat_prompt": _combo_repeat_prompt,
    "combination:two_responses": _combo_two_responses,
}


def check_instruction(instruction_id: str, response: str, kwargs: dict | None) -> bool | None:
    """True/False for a checked instruction, None when it cannot be checked here.

    kwargs may carry None for unused keys (the HF release does); they are
    dropped before dispatch. An unknown instruction id is unsupported (None)."""
    fn = CHECKERS.get(instruction_id)
    if fn is None:
        return None
    kw = {k: v for k, v in (kwargs or {}).items() if v is not None}
    try:
        return bool(fn(response, kw))
    except UnsupportedInstruction:
        return None


def score_response(response: str, instruction_ids: list[str], kwargs_list: list[dict]) -> dict:
    """Score one prompt. `strict` = all *checkable* instructions pass; a prompt
    whose every instruction is unsupported counts as unsupported, not passed."""
    results, unsupported = [], 0
    for iid, kw in zip(instruction_ids, kwargs_list):
        ok = check_instruction(iid, response, kw)
        if ok is None:
            unsupported += 1
        results.append(ok)
    checked = [r for r in results if r is not None]
    return {
        "per_instruction": results,
        "unsupported": unsupported,
        "inst_pass": sum(1 for r in checked if r),
        "inst_checked": len(checked),
        "strict": bool(checked) and all(checked),
    }


def _build_messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


class IFEvalEval:
    """Strict IFEval. Resumable via ifeval_progress.json (gpqa.py layout)."""

    def __init__(
        self,
        client: LLMClient,
        limit: int | None = None,
        results_dir: Path | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ):
        self.client = client
        self.limit = limit
        self.max_tokens = max_tokens
        self.results_dir = Path(results_dir) if results_dir else None
        self.progress = ProgressFile(results_dir, "ifeval")

    def load_items(self) -> list[dict]:
        from datasets import load_dataset

        ds = load_dataset(DATASET, split="train")
        items = [{"key": raw["key"], "prompt": raw["prompt"],
                  "instruction_id_list": list(raw["instruction_id_list"]),
                  "kwargs": [dict(k) for k in raw["kwargs"]]} for raw in ds]
        if self.limit:
            items = items[: self.limit]
        return items

    def evaluate(self) -> dict:
        items = self.load_items()
        done = drop_errored(self.progress.load())
        fb0 = getattr(self.client, "reasoning_fallback_count", 0)
        t0 = time.time()

        for i, item in enumerate(items):
            key = str(i)
            if key in done:
                continue
            response, err = chat_or_error(self.client, _build_messages(item["prompt"]),
                                          max_tokens=self.max_tokens)
            if err is not None:
                print(f"[ifeval] item {i} request error, recorded and retried on resume: {err}", flush=True)
                done[key] = {"correct": False, "completion_tokens": None, "capped": False, "inst_pass": 0, "inst_checked": 0, "unsupported": 0, "error_request": err}
                self.progress.save(done)
                continue
            tokens = getattr(self.client, "last_completion_tokens", None)
            scored = score_response(response, item["instruction_id_list"], item["kwargs"])
            capped = isinstance(tokens, int) and tokens >= self.max_tokens - CAP_SLACK
            done[key] = {
                "key": item["key"],
                "correct": scored["strict"],
                "instruction_ids": item["instruction_id_list"],
                "per_instruction": scored["per_instruction"],
                "inst_pass": scored["inst_pass"],
                "inst_checked": scored["inst_checked"],
                "unsupported": scored["unsupported"],
                "completion_tokens": tokens,
                "capped": capped,
            }
            self.progress.append_token_row({"idx": i, "completion_tokens": tokens,
                                            "correct": scored["strict"], "capped": capped})
            if (i + 1) % 25 == 0 or i + 1 == len(items):
                self.progress.save(done)
                strict = sum(1 for r in done.values() if r["correct"])
                rate = (i + 1) / max(time.time() - t0, 1e-9)
                print(f"[ifeval] {i+1}/{len(items)} prompt-strict so far "
                      f"{strict/len(done):.1%} ({rate:.1f} q/s)")

        n = len(done)
        strict = sum(1 for r in done.values() if r["correct"])
        inst_pass = sum(r["inst_pass"] for r in done.values())
        inst_checked = sum(r["inst_checked"] for r in done.values())
        unsupported = sum(r["unsupported"] for r in done.values())
        prompt_acc = strict / n if n else 0
        inst_acc = inst_pass / inst_checked if inst_checked else 0
        print(f"\n[ifeval] prompt-strict {prompt_acc:.1%} ({strict}/{n}), "
              f"inst-strict {inst_acc:.1%} ({inst_pass}/{inst_checked}), "
              f"{unsupported} unsupported instructions")
        results = {
            "score": round(prompt_acc * 100, 2),
            "metric": "prompt_strict_acc",
            "correct": strict,
            "total": n,
            "parse_failures": 0,
            "request_errors": sum(1 for r in done.values() if r.get("error_request")),
            "inst_strict_acc": round(inst_acc * 100, 2),
            "inst_pass": inst_pass,
            "inst_checked": inst_checked,
            "unsupported_instructions": unsupported,
            **token_summary(done.values(), self.max_tokens),
            "reasoning_fallback_count": getattr(self.client, "reasoning_fallback_count", 0) - fb0,
            "caveat": ("strict mode only; language:response_language is skipped "
                       "(unsupported) unless langdetect is importable"),
        }
        self.progress.write_detail(results)
        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run IFEval against an OpenAI-compatible server")
    parser.add_argument("--api-base", default="http://127.0.0.1:8090/v1")
    parser.add_argument("--model", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--think", action="store_true", help="leave reasoning on (default off)")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = parser.parse_args()

    with LLMClient(args.api_base, args.model, think=args.think) as client:
        IFEvalEval(client, limit=args.limit,
                   results_dir=Path(args.results_dir) if args.results_dir else None,
                   max_tokens=args.max_tokens).evaluate()
