# units orbit: pre-registered design and scoring

written 2026-11-05 (Europe/Paris), BEFORE any real-model output exists for this task set: no units task has been run by any model. Only the scripted mock (`mock_units.py`, no model) has driven the driver, scorer and renderer, to prove the plumbing. The task set was frozen first: `FROZEN.sha256` hashes `tasks_def.py`, `schemas.json`, `tasks.json`, `models.json` and this file. The queue item re-checks them and `freeze.py --check` at every DRYRUN and refuses to run if anything changed. The code (driver, classifier, scorer, bridge, renderer) is hashed separately in `CODE.sha256`; a harness bug found by the smoke may change code (and bumps `PARSER_VERSION` in `ucommon.py`), never the frozen files.

## the question

When the user states a quantity in one unit and a tool argument is defined in another, does a local model convert correctly, or does it slip by a power of ten? In a chat answer a slip is a wrong number. In a tool argument it is a wrong action: a 90-second timeout becomes 25 hours. arXiv 2609.25009 (one author, submitted 2026-07-27) measured numeric invariance in answers (fraction, scientific notation, percent, words, unit conversion) and found the unit failure confined to one model (Mistral Small 4, .699, 265 power-of-ten errors; Qwen3.5 .997-.999). Nobody in that paper tests units inside tool arguments. This does.

Under FP2 the subject is the MODELS. The tool schemas are the real ones Hermes ships; they name the unit in the argument description. Nothing here is framed as a lack in Hermes.

## the tasks (`tasks.json`, built by `freeze.py` from `tasks_def.py` + `schemas.json`)

40 tasks, 160 forms (prompts). Each task = one tool + one quantity argument whose unit the schema names + an exact rational expected value in that unit. Each task is written in 4 forms, only the quantity phrase changes:

| form | what it is |
|---|---|
| canonical | the argument's own unit ("90 seconds") |
| sub | a smaller unit ("90000 milliseconds", "350 centimetres", "250000 milligrams", "2500000 microseconds") |
| larger | a bigger unit or a decimal of one ("1.5 minutes", "1.2 kilometres", "0.25 kilograms", "49.99 dollars") |
| words | number words, or a fraction of a larger unit ("a minute and a half", "a fifth of a minute", "a twelfth of an hour") |

`freeze.py` proves for all 160 forms that stated number x unit factor == expected, exactly (Fractions), and refuses to freeze otherwise. Of the 120 non-canonical forms, 108 state a number that differs from the expected argument by a factor of 10 or more (94 of them by 10 to 1000x); the 12 that do not are the fraction-of-a-unit forms. Prompts never name the experiment.

REAL Hermes tools (24 tasks), schemas taken verbatim from the pinned checkout (9bda188) by `extract_schemas.py` (file + line recorded in `tasks.json` per tool):

- `terminal.timeout`, seconds, integer (`tools/terminal_tool.py` schema L1554, arg L1569): "Max seconds (default 180, foreground max 600). ..." 8 tasks.
- `process_manage.timeout` (the schema's registered name; the file is `process_registry.py`), action `wait`, seconds, integer (`tools/process_registry.py` L2582 / L2615): "Max seconds for 'wait'." 4 tasks.
- `browser_cdp.timeout`, seconds, number (`tools/browser_cdp_tool.py` L345 / L396): "Timeout in seconds (default 30, max 300)." 4 tasks.
- `browser_exec.timeout_s`, seconds, integer (`tools/browser_use_cli.py` L965 / L975): "Max seconds to wait for the code to finish (default 300, max 1800)." 4 tasks. The tool description is the static fallback of that checkout (the managed CLI is not installed there); the argument text is the live one.
- `computer_use.seconds`, action `wait`, seconds, number (`tools/computer_use/schema.py` L188 / L150): "wait: seconds to pause (max 30)." 4 tasks.

SYNTHETIC tools (16 tasks), written for this experiment, labelled synthetic in every result row and report (the model sees no such label): `charge_card.amount_cents` (cents), `trim_cable.length_m` (metres), `set_delay.delay_ms` (milliseconds), `weigh_out.mass_g` (grams); 4 tasks each. Their schema argument text names the unit in the same style as the real ones ("..., in cents.").

The real Discord tool has a minutes argument (`auto_archive_duration`, an enum) and is NOT used: an enum of four values is a multiple choice, not a conversion.

## run protocol

- one single-turn request per (model, task, form) to llama-server `/v1/chat/completions`, `tools` = the task's own tool only (one tool in the request), `tool_choice` left at the server default, system prompt "You are a helpful assistant. Use the provided tool to carry out the user's request.", one request in flight.
- greedy exactly as the paper: temperature 0, top_k 1, top_p 1, min_p 0, seed 20260725. `cache_prompt` false.
- server: llama.cpp 839a572a4 (build 11777), `--jinja` (the model's own embedded chat template), `--parallel 1`, `--ctx-size 8192`, flash-attn on, f16 KV, port 8198, never :8090. Donald is drained and restored by night-lib; one fresh server per model leg.
- thinking OFF through each model's documented switch: `chat_template_kwargs {"enable_thinking": false}` for Qwen3.8 / Ornith-1.5 / Qwen3.5. gpt-oss-20b has no off switch: `{"reasoning_effort": "low"}` and it is LABELLED "low effort, not off" everywhere. max_tokens 512 (1024 for gpt-oss, whose reasoning channel is generated before the call).
- legs in order: Qwen3.8-27B Q6_K (house model), Ornith-1.5-35B Q4_K_M, gpt-oss-20b MXFP4 (ggml-org @ c9a07b97, the revision the paper's manifest records), Qwen3.5-9B Q4_K_M, Qwen3.5-9B Q8_0 (unsloth @ 3885219b, the paper's exact files, sha256 equal to the paper's `model_hashes.json`). 160 calls per model, 800 in total. Resumable per (model, task, form).
- OPTIONAL last leg (bridge): 100 unit-view + 100 fraction-view prompts from the paper's frozen `views_benchmark.jsonl` on Qwen3.5-9B Q4 and Q8 with the paper's exact sampling (cap 1024, thinking off), to check that our runtime reproduces their per-item answers. Selection rule, fixed here: shuffle all 8,600 views with `random.Random(20260725)` as the paper does and take the first 100 `unit` views and the first 100 `fraction` views. Reported as an agreement rate against the paper's `eval.jsonl` (category and parsed value, per item). It is a runtime-parity check, not a result about tool arguments, and only runs if at least 25 minutes of the block remain after the main legs.

## scoring (`score.py`, stdlib, exact rationals; arguments are parsed with `Fraction`, never floats)

Per call exactly one of:

- `ok`: the first tool call is the right tool (and, for process/computer_use, the `wait` action), the argument is a number (a numeric string counts, flagged) and equals the expected value exactly.
- `magnitude_slip`: the value times 10^k equals the expected value for some k with 0 < |k| <= 6. k is recorded. (k = -3 means the model's value is 1000 times too big.)
- `other_value`: a number that is neither (includes negative values and 10^7 and beyond).
- `no_call`: no tool call in the reply.
- `wrong_tool`: a different tool, or the wrong action for process / computer_use.
- `malformed`: the arguments are not valid JSON, the argument is missing, or it is a non-numeric string / boolean / list.

Per task per model: invariant = all 4 forms give the same value; correct = all 4 are `ok`. Paired comparison Qwen3.5-9B Q4_K_M vs Q8_0: exact two-sided McNemar on per-call correctness (`ok` or not), over the 160 shared calls, descriptive only (one checkpoint pair, no direction pre-registered; the paper's own re-scored Q4/Q8 effect has no direction).

## the pre-registered verdict words (applied by `score.py`, never by hand)

Per model, over its 160 calls, in this order:

1. **BROKEN**: more than 20% of its calls are `no_call` or `malformed`. That is a harness or template problem, not a unit finding. The model is declared invalid and excluded from the overall verdict.
2. **SLIPS FOUND**: at least 3 `magnitude_slip` calls.
3. **CLEAN**: 0, 1 or 2 `magnitude_slip` calls.

Overall: **INVALID** (every model that ran is BROKEN), **SLIPS FOUND** (at least one valid model is SLIPS FOUND), **CLEAN** (every valid model is CLEAN). PROVISIONAL when any model has fewer than 160 calls or a leg has not run; the rule is applied to what exists and the readout says so.

## LINE 1 variants (filled from the data by `score.py`, nothing worded by hand afterwards)

- SLIPS FOUND: `asked for <the ask of the worst slip>, <model> set <the value, humanised in the argument's own unit ladder>.` plus the prompt's own wording and the slip count in brackets. The worst slip is the one with the largest |k|, ties broken by: value above the expected one first, then the earlier leg, then the task id, then form order. Example of the shape: "asked for a 90-second timeout, it set 25 hours".
- CLEAN: `local models convert units into tool arguments cleanly when the schema names the unit.` plus the counts in brackets.
- INVALID: no line 1, no post; a harness note in the vault only.

## what the readout may and may not claim (written now)

- It tests single-turn conversion into ONE argument, with ONE tool offered, greedy, thinking off (gpt-oss low effort), on 40 small tasks of one author's design. It is not a Hermes session test and says nothing about how Hermes behaves.
- 24 of 40 tasks are seconds-valued Hermes arguments; the other 16 are synthetic. The synthetic schemas are the author's wording, which may state the unit more plainly than some real ones. Real-tool and synthetic results are reported separately.
- A slip count on 160 calls per model is small. 3 slips is the action threshold, not a significance level. No ranking of models is claimed from it.
- "Nobody tests this" and "first" are not claimed: NUMCoT and Bui et al. 2025 are in the paper's related work; the paper itself tests answers, not tool arguments, and that is the only claim.
- Q4 vs Q8 is one checkpoint pair. No sentence of the form "quantization changes how a model reads a unit" is written from this.
- The paper's numbers (unit view .699 for Mistral Small 4, 265 power-of-ten errors, 8,600 prompts) are cited as the paper's, from its text and ancillary data; none is ours.

## smoke runs are not data

Before the queue run, a short GPU smoke (3 tasks x 4 forms, Qwen3.5-9B Q4_K_M, own results directory `smoke/`) checks the server, the template, the tool-call parser and the pace. Its rows are never counted, never copied into the run's `results/`, and the same tasks are run again from scratch in the real run. A harness bug the smoke finds is fixed in the harness; the models, the rule, the verdict words and the quantities stay as frozen.

Smoke finding recorded (2026-11-05 10:03 CET, Qwen3.5-9B Q4_K_M, t01+p01+u01, 12 calls, not data): the real process tool registers as `process_manage`, not `process`, so the first freeze expected a tool name no model could call (4 of 4 process calls would have scored wrong_tool). That is a task-spec bug on our side, not a model result. `freeze.py` now takes the expected tool name from the schema's own `name` field for every real tool; no quantity, prompt, expected value, model, rule or verdict word changed, and the freeze was redone before any counted output. The sha256 in the vault build log is the one AFTER this fix.

## honest limits (written before any number)

single greedy pass per cell; five models, one of them at low reasoning effort rather than off; the Qwen3.5 pair is the paper's exact files on a different runtime (llama.cpp 839a572a4 on an RTX 5090 vs the paper's b10107 on an H100), so per-item parity is checked (bridge leg), not assumed; synthetic tools are the author's; the forms are one author's phrasings; `computer_use` and `process` need an `action` the prompt does not name (the model must pick `wait`), so a `wrong_tool` there is an action error and not a unit error and is counted separately; thinking-on behaviour is not measured.
