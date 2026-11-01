# pheromone: arXiv 2601.08129 replayed on one RTX 5090

Queue item and helpers behind [reports/pheromone-replay.md](../reports/pheromone-replay.md).

- `05-pheromone-pf.sh`: the capsule idle-queue item. Builds nothing; runs the paper's `schedule-experiment` binary (Govcraft/pressure-field-experiment @ d3b0d5b, plus a second build of the same commit with the ablation list cut to full + no_decay) over 16 cells, one `grid`/`ablation` call per cell, resumable per cell, serving the Qwen2.5 chain on Ollama and Ornith on llama-server.
- `pheromone_summary.py`: per-cell solved/N, Wilson 95% interval, mean ticks, mean tokens; `--match` computes the token-matched chat tick budget.
- `pf-e2e.sh`: the one-trial-per-cell end-to-end test run before the real block.

Paths are the capsule's; adapt `E=` and the model paths to your box.
