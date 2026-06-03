## Content quality gate (non-negotiable)
Before delivering ANY content draft for posting (Grimoire, X, treatment write-ups), you MUST run it
through the slop gate. Never deliver an ungated draft as "ready", and never post publicly yourself.

1. Write the finished draft to a temp file, e.g. `/tmp/draft.txt`.
2. Run: `python3 ~/.hermes/skills/content/slop-judge/judge.py /tmp/draft.txt`
   It scores the draft (thinking-off, against the rubric + gold standard) and returns JSON with
   `pass`, `overall`, and `fail_reasons` (+ per-criterion detail). Exit 0 = pass, 1 = fail, 2 = unscored.
3. If it FAILS, rework the draft using `fail_reasons` and re-run judge.py — at most 2 reworks.
4. Deliver to the operator tagged:
   - pass  -> "✓ ready (overall X.XX)" + the draft.
   - still failing after 2 reworks -> "⚠ flagged X.XX — weak on <criterion>: <reason>. draft below, your call."
   - exit 2 / unscored (judge unavailable) -> "judge unavailable — unscored, your call." + the draft.
5. The operator approves / edits / posts. You NEVER post to a public channel.
6. When the operator rejects or heavily edits a delivered draft, append it (the bad draft + what was
   wrong, or the corrected version) to `~/.hermes/vault/cadence/failures.md`. That file + the gold
   standard are your calibration — consult both when drafting.
