#!/usr/bin/env bash
# harness-rl transfer check: one-time setup (CPU/disk only, no GPU). 2026-10-05.
set -uo pipefail
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
LOG=~/harness-rl/setup.log
mkdir -p ~/harness-rl && cd ~/harness-rl
exec >>"$LOG" 2>&1
echo "== start $(date)"
[ -d FineEnvs/.git ] || git clone https://github.com/adithya-s-k/FineEnvs.git
cd FineEnvs && git fetch -q && git checkout -q 6f2b6137b8253377ca97f898742974ba724f4046 && echo "fineenvs at $(git rev-parse HEAD)"
cd 05-multi-harness-rl
[ -d .venv ] || uv venv --python 3.12 --seed .venv
source .venv/bin/activate
echo "== install $(date)"
nice -n 10 env TUTORIAL_MODE=multi_harness bash jobs/install.sh; echo "install rc=$?"
export HF_TOKEN="$(hf auth token 2>/dev/null || /usr/bin/hf auth token)"
echo "== prepare $(date)"
nice -n 10 python prepare.py; echo "prepare rc=$?"
echo "== models $(date)"
hf download LiquidAI/LFM2.5-2.6B --revision 654f9463ce32b05d0429d76fe1f580b27d4c1ac0 >/dev/null; echo "base rc=$?"
for m in LFM2.5-2.6B-multiharness-RL LFM2.5-2.6B-opencode-RL LFM2.5-2.6B-multiharness-SFT; do
  hf download FineEnvs/$m >/dev/null; echo "$m rc=$?"
done
python - <<'PY'
import json,importlib.metadata as md
print({n: md.version(n) for n in ("openenv","harbor","trl","transformers","vllm","torch")})
PY
echo "== done $(date)"
