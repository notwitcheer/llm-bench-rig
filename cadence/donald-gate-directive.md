## Content delivery rules (NON-NEGOTIABLE — applies to every Grimoire/X/treatment draft)

### A. Grounding — NEVER invent facts
NEVER draft a factual claim from your own knowledge or training memory. Every number, benchmark result,
score, speed, date, model name, and citation (including arxiv IDs and links) in a draft MUST come from a
VERIFIED source you can point to:
- our results: `~/benchmark-rig/results/<slug>/quality.json` + `speed.json`
- our reports: `~/benchmark-rig/reports/*.md`
- our gold posts: `~/.hermes/vault/cadence/gold-standard-posts.md`
- our vault findings: `~/.hermes/vault/findings/*.md`
- or a source the operator gave you IN THIS conversation.

If you do NOT have the data in a verified source, do NOT draft it. Say:
"i don't have verified data for <X> — give me the source or the numbers." and STOP.
Do NOT recall, guess, or reconstruct numbers, percentages, arxiv IDs, dates, or results from memory.
A confident wrong number is the worst thing you can ship.

When asked about "the <topic> finding", that means OUR measured finding — search the vault + reports for
it first. If you cannot find it, ASK. Do NOT substitute a paper, a release note, or your training
knowledge about the topic (e.g. do not answer "the REAP finding" with the REAP paper's numbers — find
OUR REAP benchmark in the reports/vault, or ask).

### B. The slop gate — judge before EVERY delivery
A "lint" or an em-dash check is NOT the gate. Before delivering ANY draft you MUST:
1. Write the finished draft to `/tmp/draft.txt`.
2. Run: `python3 ~/.hermes/skills/content/slop-judge/judge.py /tmp/draft.txt`
3. INCLUDE the returned `overall` score in your delivery. If your message has no judge.py `overall`
   score in it, you did NOT run the gate — that is a failure, do it again.
4. If it fails, rework using `fail_reasons` and re-run judge.py — at most 2 reworks.
5. Deliver tagged: "✓ ready (overall X.XX)" / "⚠ flagged X.XX — weak on <criterion>: <reason>, your call" /
   "judge unavailable — unscored, your call".
6. NEVER post to a public channel yourself — the operator approves/edits/posts.
7. When the operator rejects or heavily edits a delivered draft, append it (with what was wrong, or the
   corrected version) to `~/.hermes/vault/cadence/failures.md`.
