# lib/evals

Chat-completion evals for the quality board. Every eval is a class with
`evaluate()` returning a dict with at least `score`, `metric`, `correct`,
`total`, `parse_failures`, `caveat`; `lib.quality._run_evals` writes the whole
dict to `results/<slug>/<task>_detail.json` and keeps `score`/`metric` in
`quality.json`. Fake-client unit tests live in `tests/`; nothing there touches
a server or downloads a dataset.

## board tasks (the five in q_avg)

| task | module | dataset | metric |
|---|---|---|---|
| mmlu | mmlu.py | cais/mmlu | acc |
| arc_challenge | arc.py | allenai/ai2_arc | acc |
| hellaswag | hellaswag.py | Rowan/hellaswag | acc |
| gsm8k | gsm8k.py | openai/gsm8k | exact match |
| humaneval | humaneval.py | openai/openai_humaneval | pass@1 |

`q_avg` is the mean of exactly these five (`lib.board.BOARD_TASKS`). It has
been that since the first row and must stay so for comparability.

## second-tier tasks (never folded into q_avg)

Requested alongside the board, each reported as its own row. Adding any of
them to a run changes nothing about `q_avg`: `lib.board.quality_average` reads
only the five-task allowlist, and `tests/test_quality_registry.py` asserts the
new names are outside it.

| task | module | dataset (HF) | items | metric | resumable file |
|---|---|---|---|---|---|
| gpqa | gpqa.py | Idavidrein/gpqa (gpqa_diamond, gated) | 198 | acc | gpqa_progress.json |
| ifeval | ifeval.py | google/IFEval (split train) | 541 | prompt_strict_acc (+ inst_strict_acc) | ifeval_progress.json |
| math500 | math500.py | HuggingFaceH4/MATH-500 (split test) | 500 | acc (last `\boxed{}`, qwen grader) | math500_progress.json |
| humaneval_plus | evalplus.py | evalplus/humanevalplus (split test) | 164 | pass@1 | humaneval_plus_progress.json |
| mbpp_plus | evalplus.py | evalplus/mbppplus (split test) | 378 | pass@1 | mbpp_plus_progress.json |

Notes:

- all four 2026-09-10 additions record per-item `completion_tokens` and a
  `capped` flag (completion within `CAP_SLACK` of `max_tokens`) so a budget-
  truncated run is visible in the detail json before rows are compared.
- ifeval: strict mode only, 25 instruction types checked natively in Python.
  `language:response_language` is recorded as unsupported when `langdetect` is
  not importable (it is not a rig dependency); the detail json carries
  `unsupported_instructions`. Default `max_tokens` 1024, think off.
- math500: default `max_tokens` 2048; grading is the vendored
  `lib/qwen_math_grader` (`math_equal`) with a normalised string-equality
  fallback. An unclosed `\boxed{` (truncated completion) is a parse failure.
- evalplus: the plus test suites import numpy, so the sandbox runs under the
  rig venv interpreter (`sys.executable`) with a 30 s timeout. humaneval_plus
  keeps humaneval.py's prompt exactly; expect scores below the matching
  humaneval row (~80x the tests). mbpp_plus shows the task text and the first
  original assert and expects a full function definition.
