# toolcall-v1: BFCL v4 tool-calling slice, raw mode (Qwen3.8 family)

Design + operator go: vault `99-personal/benchmark-rig/drafts/2026-10-15-toolcall-slice-design.md` (2026-10-15).

Raw mode: the prompt is rendered client-side with the official Qwen3.8 chat template (`artifacts/qwen38_official_chat_template.jinja`, Qwen/Qwen3.8-27B @ 1d4bf0f2, sha256 c3cf9e34...), sent as one string to llama-server `/completion` (temperature 0, one request at a time), parsed by `qwen38_xml.py`, and graded by BFCL's unchanged checkers (`bfcl-eval` 2026.3.23 = gorilla 6ea57973). The server only tokenises, runs kernels and samples, so every row (mainline or PrismML fork) gets byte-identical prompts and one parser.

Files
- `qwen38_xml.py`: render + XML tool-call parser; buckets capped / malformed_structure / malformed_value / leaked / no_call / ok (strict; a lenient decode is kept for sensitivity only).
- `bfcl_items.py`: BFCL loading, tools via `convert_to_tool(..., OPENAI_COMPLETIONS)`, runtime-only checker model config (underscore_to_dot), checker wrappers, GT canonicalisation, schema screen (schema_invalid vs wrong_call).
- `freeze_toolcall_v1.py` -> `toolcall-v1/` (frozen 2026-10-15 06:40 UTC, before any model output; `FROZEN.sha256`). S1 439, S2 67, S3 300, S4 100 = 899 single-turn requests; S5 multi_turn_base 100 phase 1 + 100 phase-2 reserve. Seed 20261015. One pre-freeze exclusion: `parallel_multiple_21` (GT passes a string for an array-typed parameter, not expressible as a schema-typed call).
- `roundtrip_test.py` -> `artifacts/roundtrip.json`: all 799 S1..S3 ground-truth call sets rendered through the template's own assistant tool_calls branch, parsed back: 0 failures (lossless, type-strict, BFCL checker valid); 18 parser negative controls pass.
- `render_prompts.py` -> `artifacts/prompts_toolcall-v1.jsonl.gz` (+ index, stats): every frozen prompt, think-on and think-off, with sha256 + token counts. Single-turn max 6,140 tokens (fits ctx 24,576 with the 16,384 budget); S5 first-turn p50 6,495 / max 8,168.
- `gguf_templates.py` -> `artifacts/gguf_templates.json`: embedded templates of the six rows. Saluki, GSQ-RCO, PQ2_0, PTQ1_0 embed the official template byte for byte; the unsloth Q6_K / UD-IQ3_XXS embed a patched one (`artifacts/q6k_embedded_template.jinja`: merged system/developer messages, `high` effort alias), which renders all 1,998 frozen prompts byte-identically to the official one.
- `driver.py`: probe / leg driver (single-turn + BFCL multi-turn loop), `probe.sh`: step-1 GPU probe (Donald drain + EXIT-trap restore), `smoke_cpu.sh`: CPU-only plumbing smoke on a 0.5B model.

Runtime on capsule: `~/toolcall-v1/` (own uv venv, tokenizer files, code copy, frozen list, artifacts). Results: `~/benchmark-rig/results/probe-toolcall-2026-10-15/`.
