#!/bin/bash
# astra-stop arm builder: apply edit, convert both arms with the identical recipe, quantise, record sha256, delete bf16 intermediates
set -uo pipefail
cd /home/witcheer/astra-stop
LL=/home/witcheer/llama.cpp-qwen4
PY=/home/witcheer/unsloth-env/bin/python
say() { echo "[astra-build] $(date +%H:%M:%S) $*"; }
free_gb() { df -BG --output=avail / | tail -1 | tr -dc 0-9; }
[ "$(free_gb)" -ge 180 ] || { say "ABORT: only $(free_gb) G free"; exit 2; }
mkdir -p gguf out
if [ ! -f edited/DONE ]; then
  say "apply.py (author file, pinned base rev)"
  $PY astra/apply.py --model base --out edited > out/apply.log 2>&1 || { say "ABORT apply failed"; tail -5 out/apply.log; exit 2; }
  touch edited/DONE
fi
sha256sum edited/model-00017-of-00018.safetensors base/model-00017-of-00018.safetensors > out/shard17.sha256
cat out/shard17.sha256
for arm in base edited; do
  src=$arm; [ $arm = base ] && src=base
  for q in Q4_K_M; do
    [ -f gguf/$arm-$q.gguf ] && continue
    if [ ! -f gguf/$arm-bf16.gguf ]; then
      say "convert $arm -> bf16 gguf"
      $PY $LL/convert_hf_to_gguf.py $src --outtype bf16 --no-nextn --outfile gguf/$arm-bf16.gguf > out/convert-$arm.log 2>&1 || { say "ABORT convert $arm"; tail -5 out/convert-$arm.log; exit 2; }
    fi
    say "quantise $arm $q"
    $LL/build/bin/llama-quantize gguf/$arm-bf16.gguf gguf/$arm-$q.gguf.part $q 12 > out/quant-$arm-$q.log 2>&1 || { say "ABORT quant $arm $q"; tail -5 out/quant-$arm-$q.log; exit 2; }; mv gguf/$arm-$q.gguf.part gguf/$arm-$q.gguf
  done
  rm -f gguf/$arm-bf16.gguf; say "deleted gguf/$arm-bf16.gguf"
done
sha256sum gguf/*.gguf > out/gguf.sha256; cat out/gguf.sha256
ls -l gguf
say "BUILD-DONE (safetensors dirs base/ edited/ still on disk; delete after sha record)"
