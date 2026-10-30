# agentknew: "The Agent Knew" (arXiv 2610.17301) on one RTX 5090

Scripts used to replay the paper's gpt-oss-20b pipeline on a consumer card. The paper's own code does the work
([AryaFayyazi/Honesty](https://github.com/AryaFayyazi/Honesty) @ `7dada5c`, scripts 56, 154, 155 unchanged); these files
set it up, run it and compare the result. Write-up: [`reports/agent-knew-gptoss20b.md`](../reports/agent-knew-gptoss20b.md).

| file | what it does |
|---|---|
| `prep.sh` | venv (transformers 5.14.1, torch cu130, `kernels>=0.15.2,<0.16`), gpt-oss-20b MXFP4 weights, CAPE-Bench from HF. Fails if `kernels` is outside transformers' range, because MXFP4 then dequantises silently to ~42 GB bf16 |
| `smoke.sh` + `smoke.py` | loads gpt-oss-20b through the paper's loader, checks it stayed MXFP4, captures 5 residual depths, times a 1,583-token prefill |
| `prep_data.py` | rebuilds the rollout parquets the GitHub archives omit (files over 5 MB) from the HF parquet, samples rendered token lengths |
| `replay.sh` | scripts/56 feature extraction over 18 sets (4 confirmation x- sets limited to `outputs/v57/conf_episodes.parquet`, 3 confirmation x+ sets, 11 development sets), `max_tokens=6000` |
| `watch.sh` + `analysis.sh` | after the replay: scripts/154 `stage=fit actor=gptoss20b`, scripts/155 `models=[gptoss20b]`, then `compare.py` |
| `compare.py` | ours vs the paper's `summary.json`, `gate_fit_gptoss20b.json` and per-action `scores.parquet` |
| `early_xfit.py` | provisional 2-fold cross-fit on the confirmation units only (not the paper's protocol) |
| `skipcount.py` | counts episodes and actions skipped by the 6,000-token cap |

Capsule specifics: `smoke.sh` and `replay.sh` source a local `night-lib.sh` (stops and restores the box's llama-server so
the card is free) and check that no other GPU job is running. The only edit to the paper's clone is `local_path` in
`configs/models/gpt_oss_20b.yaml`. Results: [`data/agent-knew/`](../data/agent-knew/).
