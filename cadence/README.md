# Cadence — slop gate (LLM-as-judge)

A quality gate that scores WITCHEER content drafts 0–1 against a written rubric + gold standard, so
nothing hollow/hype/unsourced ships. The judge model is Donald's on-box Qwen3.6-27B (via the live
Hermes server, thinking-off); the decision is deterministic.

## How it's used (the reliable-drafter model)
The drafter — a **Claude session or the operator** — runs the gate on a finished draft **before posting**.
We do NOT auto-draft public content with Donald: in testing it hallucinated confident citations and
skipped the gate. Donald stays the on-box co-pilot (brief/news/recall, grounding-ruled); the public
content is drafted by the reliable drafter and gated by this tool.

## Run it
```bash
# on capsule (judge = the live Hermes server):
python3 ~/.hermes/skills/content/slop-judge/judge.py /path/to/draft.txt
# -> JSON {pass, overall, fail_reasons, criteria}; exit 0 pass / 1 fail / 2 unscored
```
Pass = all 5 criteria ≥ 0.7 (meta is a hard gate). If it fails, revise using `fail_reasons` and re-run.

## Pieces
- `threshold.py` — deterministic pass/fail from the judge JSON (6 tests in `tests/test_threshold.py`).
- `slop-judge/SKILL.md` — the rubric + strict-JSON output spec (installed `~/.hermes/skills/content/slop-judge/`).
- `judge.py` — model call (thinking-off) → `threshold.py`. The tool you run.
- `gold-standard-posts.md` — calibration: the posts that landed (→ `~/.hermes/vault/cadence/`).
- `vault-findings-5090-treatments.md` — verified numbers, so drafts ground on OUR data (→ `~/.hermes/vault/findings/`).
- `judge_smoke.py` — validates the judge discriminates (held-out good draft passes ~0.89, slop fails 0.0).
- `donald-gate-directive.md` — Donald's grounding rule (in SOUL).

## Rubric (each 0–1; threshold 0.7; meta = hard gate)
ev_plus · sourced · finding · voice · meta ("would a builder bookmark/cite this?").

## Status / next
Gate works as a tool (validated). Wiring it into Donald's *autonomous* drafting failed (he won't follow
the directive / hallucinates) — hence the reliable-drafter model. The publishing **cadence** (daily
Grimoire / weekly benchmark) is sub-project 2: schedule a **Claude** session to draft (grounded) + gate +
hand to the operator to post. Deferred (v2 eval items: regression suite, production monitoring).
