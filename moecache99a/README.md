# item 99a: llama.cpp MoE expert cache vs plain `--n-cpu-moe` offload, Ornith-1.5-122B, Hermes Agent, one RTX 5090

ran Sun 2026-11-08 12:56-13:21 CET on one RTX 5090 (32 GB) + 59 GB RAM. verdict **FASTER**, ratio 1.61 (pre-registered rule, see `PREREG.md`). result section below; the run's summary files are in `run/` (paths made relative).

- `99a-moecache-race.sh`: the queue item (header = question, arms, card time, window rule). `PREREG.md` + `FROZEN.sha256` + `SEED.sha256`: the frozen question, rule and sha256s; the item refuses to run when any differs. `mc_summary.py --selftest` proves the twelve verdict branches.
- `mc_take.sh` one recorded Hermes take (own HERMES_HOME, counting proxy, pty recorder), `mc_render.py` result card + race clip, `mc_facts.py` card numbers from the run's own summary, `mc_mkhome.py` the take's Hermes config, `rp_*.py` shared helpers from item 89 (`toolcall/ripple89/`), `rp_gate.py` the pass test, `seed/` + `prompts/` the planted-bug repo and task (item 89's, byte-identical).
- the llama.cpp build is `a2b301831` (master 2026-11-08, tag b12201) with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120`; `moe-cache-tune.py` is the PR's own tuner at that commit (sha256 in `FROZEN.sha256`), not vendored here.
- `freeze.py` + `PREREG.md.tpl` regenerate `PREREG.md` / `FROZEN.sha256` / `SEED.sha256` (used once, at staging).
- the item script in this folder differs from the queued capsule copy only in header comments (internal routing lines removed); code lines are identical.


## result (run Sun 2026-11-08, 12:56-13:21 CET)

pre-registered verdict: **FASTER** (r = cache / plain token-weighted llama-server decode = 1.615, threshold >= 1.10). source `run/summary.json`.

- plain `--n-cpu-moe 24`: 3 of 3 takes CLEAN and PASS (5 of 5 pristine tests, tests untouched), decode **42.53 tok/s** (per take 42.5 / 42.4 / 42.7).
- MoE cache `-cmoe --moe-cache-mib 24896 --moe-cache-cpu-share 0` (the PR tuner's own recommendation on this card, `run/tune.json`, `run/tune.out`; not the fallback): 3 of 3 CLEAN and PASS, decode **68.7 tok/s** (per take 68.4 / 68.6 / 69.1).
- reported, not gating: fixed-prompt probe 45.39 vs 95.47 tok/s (`run/probe-*.json`); time to done mean 120.6 s vs 84.5 s; tool calls 15-17 in both arms; VRAM peak 28,850 vs 31,752 MiB of 32,607; GPU power mean ~174 W vs ~250 W.
- token recount: every take's recount = engine usage (`run/recount/`).
- limits: one file, one card, one task, 3 + 3 takes; the cache arm leaves almost no VRAM headroom at this cache size; built from master a2b301831 (tag b12201); no claim beyond this file and card. PR #30966 merged 2026-11-07 16:20 UTC and is in the tagged releases since b12201 (GitHub compare checked 2026-11-08).
- the queued script that ran (md5 054f159f) differs from `99a-moecache-race.sh` here in header comments only; code lines are identical.
