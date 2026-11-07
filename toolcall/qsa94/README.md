# item 94: QSA fork of Qwen3.8-27B at 255K context against stock llama.cpp, one RTX 5090

STAGED 2026-11-06, ran 2026-11-07 (result below). Tests the fork README's claim (141.8 tok/s decode at 255K on an RTX 5090; stock figure not published for that card).
`PREREG.md` is frozen by `FROZEN.sha256` (the docs text `llms-full.txt`, fetched 2026-11-06 from hermes-agent.nousresearch.com/docs, is hashed there too and is not copied here: `qa_build.py` rebuilds `docs_slice.txt` and `prefill.json` from it).
`94-qsa-255k.sh` is the capsule queue item; harness files derive from the item-89 ones.

## result (run Sat 2026-11-07 02:13-02:56 CET, one RTX 5090)

verdict on the pre-registered rule: **QSA AHEAD** (R255 = 184.9 / 82.0 = 2.25, threshold 1.5). claim check against the README's 141.8 tok/s: above.

decode tok/s, code task, cold, seeds 11 + 12 (QSA / stock): empty context 238.9 / 217.6; 64K 198.5 / 154.0; 128K 200.2 / 122.9; 255K **184.9 / 82.0**.
prefill tok/s at 255K: 2,882 / 1,117. prose task at 255K: 121.7 / 60.8. early stops under 16 tokens: 0 of 20 on both.

Hermes Agent, whole docs text as prefill (about 241K tokens), 24 one-number questions: QSA 23/24 in 386 s, stock 21/24 in 630 s, VRAM peak 23,492 / 23,886 MiB, 0 request errors.

post-run renderer fix (`c94_render.py`, second commit): the prefill race end frame showed a dash for QSA's decode rate (its 256-token tail lasts ~1 s, shorter than the 2 s rate window). the fix adds optional `--final-q` / `--final-k` so the finished lanes show the server's own timings for that run (QSA 2,887 / 192 tok/s, stock 1,117 / 85 tok/s, seed 11). without the flags the output is unchanged.
