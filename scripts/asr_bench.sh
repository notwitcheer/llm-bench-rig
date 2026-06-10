#!/usr/bin/env bash
# ASR head-to-head — capsule generation. For each model+split, transcribe every LibriSpeech
# utterance via its ggml .cpp runtime, capture transcript + wall-clock timing + peak VRAM.
# Donald paused ONCE (clean RTFx), restored on exit. Writes results/asr/<model>/{transcripts.jsonl,timing.json}.
# Assumes the pre-flight built the runtimes and staged the dataset (see the plan's pre-flight).
# Run in tmux:  tmux new -d -s asrbench "./scripts/asr_bench.sh 2>&1 | tee ~/asrbench.log"
set -uo pipefail
cd ~/benchmark-rig
PARAKEET_BIN=~/parakeet.cpp/build/bin/parakeet-cli      # confirmed path set in pre-flight
WHISPER_BIN=~/whisper.cpp/build/bin/whisper-cli
DATA=~/asr-data/librispeech                              # <split>/manifest.jsonl: {id, split, audio, duration, reference}
PY=.venv/bin/python

restore() { systemctl is-active --quiet llama-server.service || sudo -n systemctl start llama-server.service || true; }
trap restore EXIT
sudo -n systemctl stop llama-server.service; sleep 3

# transcribe one model over both splits -> transcripts.jsonl + timing.json
run_model() {  # $1=model_name  $2=transcribe_cmd_template (uses {AUDIO})
  local name="$1" tmpl="$2" out=~/benchmark-rig/results/asr/"$1"
  mkdir -p "$out"; : > "$out/transcripts.jsonl"
  local audio_s=0 t0 t1
  local before; before=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  t0=$(date +%s.%N)
  for split in test-clean test-other; do
    while IFS= read -r line; do
      local id ref audio dur hyp
      id=$($PY -c "import json,sys;print(json.loads(sys.argv[1])['id'])" "$line")
      ref=$($PY -c "import json,sys;print(json.loads(sys.argv[1])['reference'])" "$line")
      audio=$($PY -c "import json,sys;print(json.loads(sys.argv[1])['audio'])" "$line")
      dur=$($PY -c "import json,sys;print(json.loads(sys.argv[1])['duration'])" "$line")
      hyp=$(eval "${tmpl//\{AUDIO\}/$DATA/$split/$audio}" 2>/dev/null)
      audio_s=$($PY -c "print($audio_s + $dur)")
      $PY -c "import json,sys;print(json.dumps({'id':sys.argv[1],'split':sys.argv[2],'reference':sys.argv[3],'hypothesis':sys.argv[4]}))" \
        "$id" "$split" "$ref" "$hyp" >> "$out/transcripts.jsonl"
    done < "$DATA/$split/manifest.jsonl"
  done
  t1=$(date +%s.%N)
  local peak; peak=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  $PY -c "import json;json.dump({'audio_seconds':$audio_s,'proc_seconds':$t1-$t0,'peak_vram_mib':max($before,$peak)},open('$out/timing.json','w'))"
  echo "[asr] $name done"
}

run_model "parakeet-tdt-0.6b-v2"   "$PARAKEET_BIN --model ~/asr-models/parakeet-tdt-0.6b-v2 --device CUDA0 -f {AUDIO} --txt"
run_model "whisper-large-v3-turbo" "$WHISPER_BIN -m ~/asr-models/ggml-large-v3-turbo.bin -f {AUDIO} -nt -np"
run_model "whisper-large-v3"       "$WHISPER_BIN -m ~/asr-models/ggml-large-v3.bin -f {AUDIO} -nt -np"

echo "ASR_BENCH_DONE $(date +%H:%M:%S)"
