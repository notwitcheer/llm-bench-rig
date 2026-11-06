# item 89 pre-registration: Ripple-65B Preview driving Hermes Agent on one RTX 5090

frozen 2026-11-05, before any counted output (the only GPU use so far is two staging smokes: llama-bench + a tool-call request, no Hermes session and no repo task).

## question

Does Ripple-65B Preview (dense 65B, Apache-2.0), run fully local on one 32 GB RTX 5090 through the llama.cpp PR #31692 build with community GGUF files, drive a real Hermes Agent session to fix a small Python repo with 4 planted bugs, so that the pristine tests go 5/5 and the agent never edits tests/? Test of the shape of the public GB10 claim (rlaope, 2026-11-05); not a reproduction of its numbers (different box, NVFP4 + speculative decoding, unpublished repo).

## design (all frozen)

- model files: Ripple-65B-Preview-AD-IQ3_XXS.gguf (27,964,982,752 bytes, primary, 3 takes) and AD-IQ2_S.gguf (25,310,774,752 bytes, 2 takes), AtomicChat GGUFs from the BF16 weights.
- engine: llama.cpp PR #31692 branch at 8a5f7b069, --jinja, -ngl 99, flash-attn on, KV q8_0, ctx 65,536, one slot, no speculative decoding, thinking on.
- agent: Hermes pin 9bda188, own HERMES_HOME per take (never the main gateway, never :8090), toolsets terminal + file, approvals off, memory off, max 60 turns, wall cap 10 min per take.
- sampling: temperature 0.6, top-p 0.95, min-p 0.05, seed 721 through extra_body (ours; the Ripple model card gives no recommendation at fetch).
- task: prompts/fix-shopcart.txt on the seed repo (shopcart/ with cart.py paging.py pricing.py text.py, tests/ with 5 tests, 1 passing and 4 failing on the untouched seed).
- order: IQ3_XXS takes 1-3, then IQ2_S takes 1-2.

## frozen hashes (sha256; the item refuses to run when any differs)

- rp_summary.py (verdict rule + self-test): c630cb3fa3927003cb4784cf9a24a0622074742f1831e4fe01f9368489862afe
- rp_gate.py (the pass test): 6b8dcbb5194e4c6773622a1fc48254ba8bc597938c8b8e69661ce881d82b2d57
- prompts/fix-shopcart.txt: d2e53c7d08ad34e5c974a410b6c5fa22d0fae3d0a5779ec530da94ec0a223afb
- seed repo (`find . -type f | LC_ALL=C sort | xargs sha256sum | sha256sum`, run in seed/, pycache excluded): dc0a36db1200e5203034c020966724b2b2711303fed911b227622c2ee42a5014

## rule (rp_summary.py applies it as written; `rp_summary.py --selftest` proves nine branches)

- take CLEAN: ended done or wall, at least 3 tool calls in state.db, request errors at most 20% of requests.
- take PASS: CLEAN and the gate passes (pristine tests 5 passed, 0 failed, 0 errors; no file under the agent's tests/ differs from the pristine one, none added) and the agent's tool calls never named the harness dir.
- file BROKEN: did not load, tool-call smoke failed, or no CLEAN take.
- overall: BROKEN (every file BROKEN) | REPRODUCED (IQ3_XXS ran at least 2 takes and every one is a PASS) | PARTIAL (not REPRODUCED, at least one PASS on any file) | NOT REPRODUCED (clean takes, no PASS) | INCOMPLETE (fewer than 2 IQ3_XXS takes ran: provisional, the item exits 3 and resumes).
- "REPRODUCED" means the same shape of task solved fully local. It never means his numbers were reproduced and no speed multiplier is claimed.

## reported, never gated

time to done from the prompt submit, tool calls, tokens (server count vs usage), decode tok/s and prompt tok/s from llama-server timings, GPU watts (nvidia-smi 1 Hz, whole card) and peak VRAM. The 1m 36s / 14 tool calls / 44 W / 34-36 tok/s figures of the claim are quoted as his, with his hardware and decoder named.

## clip plan

take 1 of each file unless a recorder fault (then take 2), start synced on each take's Enter, 3 s cold open on the result card, real speed up to 240 s then compressed middle with a visible pill. leak read per take. an em dash in the agent's own output stays (footage is not edited).

## limits

n=3 + n=2 takes; one seed; same-shape task (4 small planted bugs), not his repo; Preview release with light post-training; community quants of an unmerged PR build; the self-reported Terminal-Bench 2.1 number (80.5) is the vendor's and is not tested here.

## addendum 2026-11-05 18:40 CET (after the freeze, before any counted output)

one pilot take of this very task ran with RP_SMOKE=1 (slug ripple89-ripple65b-hermes-repo-smoke, IQ3_XXS, 1 take): 75 s, 18 tool calls, pristine tests 5/5, tests untouched, decode 41.4 tok/s, 529 W mean. it is a pilot of the harness, not a counted take, and it is quoted as such. the frozen files above were not changed after it (hashes re-checked by the item's own plan phase). the counted run uses slug ripple89-ripple65b-hermes-repo and starts from no takes.
a pilot pass lowers the chance that the REPRODUCED branch is a surprise; it does not change the rule.
