---
name: content-draft
description: "THE way to draft any of the operator's content (X thread, X article, X quote, Telegram Grimoire piece) in his exact voice, grounded in OUR verified data, gated by the slop-judge rubric. ALWAYS use this for drafting posts; never free-style the voice."
version: 1.1.0
platforms: [linux]
metadata:
  hermes:
    tags: [content, writing, social-media, x, telegram, grimoire, witcheer]
    category: social-media
---
# Content Draft (WITCHEER voice)

## When to Use
Whenever the operator asks to draft a post, thread, X article, quote tweet, or Telegram Grimoire piece.
Format modes: `x-thread`, `x-article`, `x-quote`, `tg-grimoire` (see references/formats.md).

## Procedure
1. **Load the voice spec** - read references/voice.md and follow it EXACTLY (lowercase except brands + "I",
   UK English, practitioner tone, no hype words). Read references/formats.md for the chosen mode's structure.

2. **Ground in OUR data FIRST (anti-hallucination — non-negotiable).**
   If the topic is one of OUR benchmarks / findings / results, you MUST read our verified data BEFORE drafting:
   - `~/.hermes/vault/findings/` (e.g. `5090-treatments.md`) and `~/benchmark-rig/reports/*.md`
   - `~/.hermes/vault/cadence/gold-standard-posts.md`
   - then the `vault` skill for any other operator context.
   "the <topic> finding" means OUR measured result. NEVER substitute a paper, arxiv entry, release note,
   or your training knowledge for our benchmark. (Example: the REAP "finding" = OUR REAP-28B vs parent-35B
   head-to-head on the 5090 — NOT the REAP paper's numbers.) If our data is not in those files, STOP and
   say "i don't have verified data for <X> — give me the source or the numbers." Do not draft from memory.

3. **Source every claim — and for OUR models, ONLY our numbers.** If the post is about a model/result WE
   benchmarked, EVERY quantitative claim (speed, VRAM, size, scores, quant) MUST be the number from our
   report/findings. Do NOT pull in web/community/spec numbers or tool/fork names from `web_search` or memory
   to "enrich" it — not to supplement, and NEVER to contradict our measurement. (Real failure to avoid:
   writing "community reports 6-12 tok/s" when our report says 47; naming a "RotorQuant fork" not in our data;
   citing a Q4_K_M 78GB build when we ran MXFP4 59GB.) If a number or mechanism isn't in our data, either
   leave it out or ASK — the post is about OUR measurement, not a literature review. `web_search` is only for
   a genuinely external topic the operator explicitly asks you to explain; cite ONLY what you fetched; never
   fabricate a citation, arxiv ID, or statistic.

4. **Add insight, not number-swaps** - lead with genuine findings (what is surprising/useful). The card
   carries the numbers; the copy adds context, comparison, the mechanism, practical takeaways.

5. **Offer a card** - for benchmark/finding posts, offer or produce a Gold & Crimson card via the
   `gold-crimson-card` skill; return the PNG path with the draft.

6. **Style gate — anti-slop lint (MANDATORY, deterministic).** Write the finished draft to /tmp/draft.txt and run:

       python3 ~/.hermes/skills/social-media/content-draft/scripts/slop_lint.py /tmp/draft.txt

   Fix EVERY flagged line, then re-run until it prints "SLOP-LINT: clean" (exit 0). Read references/anti-slop.md.
   This catches AI style-tells (em-dashes, antithesis, scaffolding). It is NOT the quality gate — step 7 is.

7. **Quality gate — slop-judge rubric (MANDATORY).** After the style lint is clean, score the draft:

       python3 ~/.hermes/skills/content/slop-judge/judge.py /tmp/draft.txt

   It returns JSON: `pass`, `overall`, `fail_reasons`, and per-criterion scores (ev_plus, sourced, finding,
   voice, meta). Pass = all five >= 0.7 (meta is a hard gate). If it does NOT pass, rework the draft using
   `fail_reasons` and re-run (max 2 reworks). A draft that has not PASSED slop-judge is not finished.

8. **Deliver only - tagged with the score.** Output the draft (+ any card path) tagged:
   "✓ ready (overall 0.XX)" if it passed, or "⚠ flagged (0.XX) — weak on <criterion>: <reason>, your call"
   if still failing after 2 reworks. Do NOT post anything; the operator posts. Public identity = WITCHEER only.
   If the operator later rejects/edits a draft, append it (+ what was wrong) to `~/.hermes/vault/cadence/failures.md`.

## Pitfalls
- Biggest failure: drafting OUR finding from a paper or memory instead of our reports/vault (step 2). Always read our data first.
- Hype words / US spellings (step 6). Do not invent sources or stats (step 3).
- "slop-lint clean" is NOT done — you must ALSO pass slop-judge (step 7) and report its score (step 8).
