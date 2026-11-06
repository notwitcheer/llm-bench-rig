# item 89: Ripple-65B Preview driving Hermes Agent on one RTX 5090

Same-shape test of a public "fixed a broken repo fully local" claim: a real Hermes Agent session fixes a small Python library with 4 planted bugs, pass = the pristine tests go 5/5 and tests/ stays untouched.
Ripple-65B Preview (dense 65B, Apache-2.0) as community GGUF files on a llama.cpp build from PR #31692 (head 8a5f7b069; the PR merged on 2026-11-05 after this run), one 32 GB card, no speculative decoding.

- `89-ripple-65b-hermes-repo.sh`: the queue item (header = hook, EV, card time, window rule). `PREREG.md`: the frozen question, rule and sha256s. `rp_summary.py --selftest` proves the verdict branches.
- `rp_take.sh` one recorded take (own HERMES_HOME, counting proxy, pty recorder), `rp_render.py` race clip + end card, `rp_gate.py` the pass test, `seed/` the planted-bug repo, `prompts/` the task.
- the reference fix and the mock model used for the CPU proof are kept out until after the run (the agents have a terminal).
- results (run 2026-11-06 01:29-01:43 CET): pre-registered verdict REPRODUCED. IQ3_XXS 3/3 takes and IQ2_S 2/2 takes ended with the pristine tests 5 passed and tests/ untouched. per take (finish s / tool calls / decode tok/s / GPU W mean): iq3xxs-1 186.6 / 18 / 41.5 / 570.8, iq3xxs-2 92.2 / 18 / 42.0 / 565.3, iq3xxs-3 95.9 / 19 / 41.9 / 562.2, iq2s-1 53.8 / 15 / 45.1 / 559.4, iq2s-2 53.0 / 15 / 45.1 / 555.9. VRAM peak 29,808 / 27,278 MiB at ctx 65,536, KV q8_0. summary: `data/ripple89/summary.json`.
- one task and five takes: the smaller file was faster here, which is an observation, not a which-file verdict.
