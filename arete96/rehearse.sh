#!/bin/bash
# rehearse.sh: PRODUCTION-PATH rehearsal of 96-arete-eval.sh on capsule with NO card, NO Donald, NO :8090: a copy of the item with the GPU-touching lines swapped for no-ops and a shim "llama-server" that runs the CPU stub.
# usage: rehearse.sh <deadline-seconds-from-now> <run-dir> ; prints the item's own log lines. Never queued, never a result.
set -u
DLS=${1:-3000}; RUNX=${2:-/tmp/reh96/run}
R=/tmp/reh96; mkdir -p $R; rm -f $R/item.sh
S=/home/witcheer/voxel-stage/arete96
cat > $R/shim.sh <<'EOS'
#!/bin/bash
# shim llama-server: --version / --help for the PLAN checks, otherwise the CPU stub on --port
for a in "$@"; do
  [ "$a" = --version ] && { echo "version: 0.7.0-dev (build 11777, commit 839a572a4)" >&2; exit 0; }
  [ "$a" = --help ] && { echo "--jinja --n-gpu-layers --host --port --flash-attn --parallel --ctx-size --cache-type-k --cache-type-v --metrics"; exit 0; }
done
port=""; model=""
while [ $# -gt 0 ]; do case "$1" in --port) port=$2; shift;; -m) model=$2; shift;; esac; shift; done
export REH_DELAY=${REH_DELAY:-0.001}; mode=base; [[ "$model" == *arete* ]] && mode=arete
exec /home/witcheer/scratch/race/venv/bin/python /home/witcheer/voxel-stage/arete96/code/stub_server.py "$port" --items /home/witcheer/voxel-stage/arete96/data/items.jsonl --mode $mode --delay ${REH_DELAY:-0.001}
EOS
chmod +x $R/shim.sh
head -c 1000 /dev/zero > $R/base.gguf; head -c 1100 /dev/zero > $R/arete.gguf
sed -e "s#^SRV=.*#SRV=$R/shim.sh; SRV_SHA=839a572a4#" \
    -e "s#^G_BASE=[^;]*;#G_BASE=$R/base.gguf;#" -e "s#B_BASE=[0-9]*#B_BASE=1000#" \
    -e "s#^G_ARETE=[^;]*;#G_ARETE=$R/arete.gguf;#" -e "s#B_ARETE=[0-9]*#B_ARETE=1100#" \
    -e 's#^mem_used() .*#mem_used() { echo 0; }#' \
    -e 's#^FOREIGN=.*#FOREIGN=none#' \
    -e 's#DONALD_STOPPED=1; nl_donald_stop#DONALD_STOPPED=1; true#' \
    -e 's#\[ "\$DONALD_STOPPED" = 1 \] && \[ "\$DONALD_BACK" = 0 \] || return 0#return 0#' \
    -e 's#^  nvidia-smi --query-gpu=timestamp.*#  sleep 3600 \& SMP=$!#' \
    $S/96-arete-eval.sh > $R/item.sh
if [ "${REH_RELAX:-0}" = 1 ]; then sed -i -e 's#\[ "\$LEFT" -ge 25 \]#[ "$LEFT" -ge 1 ]#' -e 's#\[ "\$lft" -lt 15 \]#[ "$lft" -lt 0 ]#' $R/item.sh; fi
# nothing in the rehearsal copy may touch Donald or sudo
bad=$(grep -nE '^[^#]*(sudo |localhost:8090|nl_donald_stop|systemctl)' $R/item.sh | grep -v 'sudo -n systemctl start llama-server.service' || true)
live=$(grep -nE '^[^#]*(nvidia-smi)' $R/item.sh || true)
echo "REHEARSAL GUARD: donald lines left: [${bad}] nvidia-smi lines left: [${live}]"
[ "${REH_KEEP:-0}" = 1 ] || rm -rf "$RUNX"
NL_HARD_STOP_EPOCH=$(( $(date +%s) + DLS )) MANIFEST_OVERRIDE=$R/m.json ARETE_RUN_OVERRIDE=$RUNX bash $R/item.sh
echo "REHEARSAL RC=$?"
