# Cadence — slop gate (LLM-as-judge), Hermes-native

A quality gate that scores WITCHEER content drafts 0–1 against a written rubric + gold standard, so
nothing hollow/hype/unsourced/ungrounded ships. The judge model is Donald's on-box Qwen3.6-27B; the
pass/fail decision is deterministic. Fully sovereign — no cloud in the loop.

## How it works (automatic, on-box)
Donald auto-invokes his **content-draft skill** for any "draft X" request. That skill (v1.1) now:
1. **Grounds on OUR data first** — reads `~/.hermes/vault/findings/` + `~/benchmark-rig/reports/` and uses
   OUR measured numbers; "the <topic> finding" = our result, never a paper/arxiv/memory; if our data
   isn't there it ASKS instead of fabricating.
2. **Style gate** — `scripts/slop_lint.py` (AI tells: em-dashes, antithesis, scaffolding).
3. **Quality gate** — `python3 ~/.hermes/skills/content/slop-judge/judge.py /tmp/draft.txt` (the 5-criterion
   rubric: ev_plus, sourced, finding, voice, meta; pass = all ≥0.7, meta hard gate). Reworks if it fails.
4. Delivers tagged "✓ ready (0.XX)" / "⚠ flagged (0.XX)" + a Gold & Crimson card. Never posts — the operator posts.

Verified: plain prompt "draft a post about the REAP finding" → Donald grounded on our numbers, scored
0.87, delivered tagged + a card. No explicit instruction needed.

## Pieces
- `content-draft-SKILL.md` — the enhanced entry-point skill Donald uses (installed `~/.hermes/skills/social-media/content-draft/SKILL.md`). THIS is where grounding + the gate are enforced.
- `slop-judge/SKILL.md` + `judge.py` + `threshold.py` (6 tests) — the rubric judge (installed `~/.hermes/skills/content/slop-judge/`).
- `gold-standard-posts.md` — calibration (→ vault). `vault-findings-5090-treatments.md` — verified numbers (→ vault `findings/`).
- `judge_smoke.py` — validates the judge discriminates (good ~0.89, slop 0.0).
- `donald-gate-directive.md` — grounding+gate rules in SOUL (backup; the content-draft skill is the real enforcement).

## Manual use (e.g. to check any draft)
`ssh witcheer@192.168.1.9` → `python3 ~/.hermes/skills/content/slop-judge/judge.py /path/draft.txt` (pass=all ≥0.7).

## Next (sub-project 2 — the cadence)
Schedule it: a cron asks Donald (via `hermes -z`/gateway) to draft the daily Grimoire candidate + the
weekly benchmark shortlist → grounded + gated by the above → Telegram for the operator to approve/post.
Deferred v2 eval items: regression suite, production monitoring.
