#!/bin/bash
# nerve-gate prep (CPU/network only, no GPU): venv + gpt-oss-20b MXFP4 weights + CAPE-Bench dataset.
# EV: protects a possible "The Agent Knew" reproduction post (held-out AUROC + per-action gate latency on a 5090).
set -uo pipefail
W=$HOME/esoteric/nerve-gate; V=$HOME/esoteric/.venv; mkdir -p "$W"
log(){ echo "[prep] $(date +%H:%M:%S) $*"; }
free_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
[ "$free_gb" -lt 60 ] && { log "FAIL disk floor: ${free_gb}G free < 60G"; exit 2; }
log "disk ok ${free_gb}G"

UV=$(command -v uv || ls "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" 2>/dev/null | head -1)
if [ ! -x "$V/bin/python" ]; then
  if [ -n "$UV" ]; then "$UV" venv -q -p 3.12 "$V"; else python3 -m venv "$V"; fi
fi
if [ -n "$UV" ]; then PIP=("$UV" pip install -q -p "$V/bin/python"); else PIP=("$V/bin/pip" install -q); fi
log "venv $V (uv=${UV:-none})"
"${PIP[@]}" torch --index-url https://download.pytorch.org/whl/cu130 || { log "FAIL torch install"; exit 3; }
"${PIP[@]}" "transformers==5.14.1" accelerate safetensors huggingface_hub "kernels>=0.15.2,<0.16" pandas pyarrow scikit-learn numpy omegaconf || { log "FAIL deps"; exit 3; }
"$V/bin/python" - <<'PY'
import transformers, torch
from transformers.utils import is_kernels_available
import transformers.utils.import_utils as q
print("[prep] transformers", transformers.__version__, "torch", torch.__version__, "kernels_ok", is_kernels_available(),
      "kernels_range", q.KERNELS_MIN_VERSION, q.KERNELS_MAX_VERSION)
import sys; sys.exit(0 if is_kernels_available() else 5)
PY
[ $? -eq 0 ] || { log "FAIL kernels outside transformers range: MXFP4 would silently dequantise to bf16"; exit 5; }

log "download gpt-oss-20b (MXFP4, no original/ or metal/)"
hf download openai/gpt-oss-20b --exclude "original/*" --exclude "metal/*" --local-dir "$HOME/models/gpt-oss-20b" >/dev/null || { log "FAIL model download"; exit 4; }
du -sh "$HOME/models/gpt-oss-20b"
log "download CAPE-Bench dataset"
hf download Afayyazi/CAPE-Bench --repo-type dataset --local-dir "$HOME/esoteric/cape-bench" >/dev/null || { log "FAIL dataset download"; exit 4; }
du -sh "$HOME/esoteric/cape-bench"
log "PREP DONE"
