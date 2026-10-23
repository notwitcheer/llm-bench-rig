#!/usr/bin/env bash
# Humming vs Marlin on the 5090: install (NO GPU). new venv, stock PyPI vLLM 0.32.0 (pins humming-kernels[cu13]==0.1.17).
set -u
ENV=/home/witcheer/humming-env
UV=/home/witcheer/.local/bin/uv
export MAX_JOBS=4 NVCC_THREADS=1 UV_CONCURRENT_BUILDS=1
say(){ echo "[hmI] $(date +%T) $*"; }
cd /home/witcheer
[ -d "$ENV" ] || { $UV venv --python 3.12 "$ENV"; say "venv rc=$?"; }
say "vllm==0.32.0 from PyPI"
timeout 1500 nice -n 19 $UV pip install --python $ENV/bin/python "vllm==0.32.0"; say "vllm rc=$?"
$ENV/bin/python - <<'EOF'
import torch, vllm, importlib.metadata as md
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_arch_list())
print("vllm", vllm.__version__)
for p in ["humming-kernels","flashinfer-python","transformers","triton","nvidia-cuda-nvcc","nvidia-cuda-nvrtc","compressed-tensors"]:
    try: print(p, md.version(p))
    except Exception as e: print(p, "MISSING")
import humming; print("humming at", humming.__file__)
EOF
say "import rc=$?"
df -h / | tail -1
du -sh $ENV
say "INSTALL_DONE"
