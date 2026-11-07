# arete96: Arete-27B vs Qwen3.8-27B (Q6_K) on Nous Research's own Propaganda_Eval

Queue item `96-arete-eval.sh` (staged and run 2026-11-07). It runs the open **Censorship & Propaganda Eval** (Hugging Face dataset `NousResearch/Propaganda_Eval`, pinned revision
`401223a2bfcdcd99d4650a83c4d20565db0e56b8`, 803 prompts) on **Arete-27B** (Nous Research's SFT of Qwen3.8-27B) and on its base **Qwen3.8-27B**, both as Q6_K GGUF, on one RTX 5090, with the eval's own
grader and scorer unchanged and the judge `x-ai/grok-4.7` at temperature 0. The question: does the model card's headline (any-propaganda 22% -> 0.9%, fact mean 2.32 -> 2.98 of 5, thinking off,
temperature 0) hold in the files people download, run once per file by a third party? A frozen capability board (GPQA-diamond, HumanEval, ARC-Challenge, MMLU slice) is run on the Arete file and
paired against the base file's own rows from item 90a.

`PREREG.md` is frozen before any counted output (hashes in `FROZEN.sha256`, checked by the item's plan phase). The verdict rule (HOLDS / SHRINKS / PARTIAL / BROKEN / INCOMPLETE) is applied by
`code/arete_summary.py`; `--selftest` proves every branch and both sides of every threshold (27 cases). Facts and counts only. Limits are in `PREREG.md`: one run per file, 4 requests in flight at
temperature 0, different GGUF cutters (Unsloth vs bartowski) with different imatrix data, Q6_K only.

## layout

| path | what |
|---|---|
| `96-arete-eval.sh` | the idle-queue item: PLAN phase (frozen hashes, build, CPU proofs incl. the whole pass path in mock mode), then three resumable GPU legs: board on the Arete file, 803 base answers, 803 Arete answers |
| `code/gen_eval.py` | generation client: streamed, 4 in flight, per-chunk timestamps saved, final-answer-only rule, append-only and resumable |
| `code/arete_summary.py` | the frozen verdict rule + paired tests + category table + the pre-registered clip prompt pick; `--selftest` |
| `code/render_arete.py`, `render.sh` | CPU-only clip: result card, then a two-pane replay of the picked prompt at the real stream speed ending on the judge verdicts |
| `code/run_legs.py`, `code/nerf_common.py`, `code/nerfevals/` | the board code of item 90a, byte-identical (hashes in `FROZEN.sha256`) |
| `ref90a/*.jsonl` | item 90a's per-item rows for the base file (same build, same code): the comparator |
| `code/stub_server.py`, `code/mock_support.py`, `rehearse.sh` | the CPU proofs (stub servers and a stub judge, never a result) |

## reproduce

```bash
# 1. fetch the pinned eval files (MIT, unchanged) next to the code
R=401223a2bfcdcd99d4650a83c4d20565db0e56b8
mkdir -p data judge
curl -sL https://huggingface.co/datasets/NousResearch/Propaganda_Eval/resolve/$R/data/items.jsonl -o data/items.jsonl
for f in JUDGE_PROMPT.md grade.py score.py; do curl -sL https://huggingface.co/datasets/NousResearch/Propaganda_Eval/resolve/$R/judge/$f -o judge/$f; done
sha256sum -c FROZEN.sha256            # every line must say OK

# 2. generate (needs llama.cpp llama-server and the two GGUFs; paths are set at the top of 96-arete-eval.sh)
#    server line used for both files: --jinja -ngl 99 --flash-attn on --parallel 4 --ctx-size 131072 --cache-type-k q8_0 --cache-type-v q8_0 --metrics
python code/gen_eval.py --port 8080 --tag base  --items data/items.jsonl --out run   # thinking off, temperature 0, max_tokens 30000, streamed
python code/gen_eval.py --port 8080 --tag arete --items data/items.jsonl --out run   # (restart the server with the other file first)

# 3. grade with the eval's own grader, unchanged (any OpenAI-compatible endpoint serving x-ai/grok-4.7)
export JUDGE_BASE_URL=https://your-endpoint/v1 JUDGE_API_KEY=...
for t in base arete; do python judge/grade.py --completions run/completions-$t.jsonl --out run/grades/grades-$t.jsonl; done

# 4. verdict (also cross-checks the numbers against judge/score.py)
python code/arete_summary.py run
python code/arete_summary.py --selftest
```

The queue item itself needs the capsule paths hard-coded at its top (`/home/witcheer/...`); the three GPU legs are resumable per row (re-run and it continues), and it exits 3 while legs remain.
The judge step is deliberately outside the item (the judge API key never goes on the GPU box).

## result so far (GPU legs Sat 2026-11-07 11:41-13:00 CET; the judge step is still grading)

capability board, Arete-27B Q6_K vs the base Qwen3.8-27B Q6_K row in `ref90a/` (same items, build and settings, greedy, thinking off), paired exact McNemar:

| task | Arete | base | p |
|---|---|---|---|
| GPQA-diamond | 104 / 198 | 98 / 198 | 0.36 |
| HumanEval | 155 / 164 | 154 / 164 | 1.0 |
| ARC-Challenge | 1,133 / 1,172 | 1,132 / 1,172 | 1.0 |
| MMLU slice (19 subjects) | 2,195 / 2,614 | 2,192 / 2,614 | 0.85 |

no measurable difference on any task. Propaganda_Eval answers: 803 / 803 (base) and 802 / 803 (Arete) generated; grades and the verdict are added when the judge step finishes.
