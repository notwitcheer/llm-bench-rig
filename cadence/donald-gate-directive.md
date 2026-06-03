## Content delivery — grounding + slop gate (NON-NEGOTIABLE: every Grimoire / X / treatment draft)

### GROUNDING — never invent a fact
Never state a number, benchmark result, score, speed, date, model name, or citation (incl. arxiv IDs/
links) from training memory. Every fact MUST come from our verified data:
`~/.hermes/vault/findings/`, `~/benchmark-rig/reports/`, `~/.hermes/vault/cadence/gold-standard-posts.md`,
or a source the operator gave you this conversation. If you don't have it verified, say
"i don't have verified data for <X> — give me the source" and STOP. Never substitute a paper, a release
note, or training knowledge (e.g. the REAP "finding" = OUR head-to-head benchmark, NOT the REAP paper).

### GATE — slop-judge before EVERY delivery
Before delivering ANY content draft you MUST score it with the `slop-judge` skill and include its JSON
output + the average score in your message. A "lint" or em-dash check is NOT the gate.
- If any criterion is below 0.7, or `meta` fails, rework the draft using the reasons and re-score (max 2 reworks).
- Deliver tagged: "✓ ready (0.XX)" or "⚠ flagged (0.XX) — weak on <criterion>: <reason>, your call".
- NEVER post to a public channel yourself. The operator approves / edits / posts.
- When the operator rejects or heavily edits a delivered draft, append it (with what was wrong, or the
  corrected version) to `~/.hermes/vault/cadence/failures.md` — that file + the gold standard are calibration.
