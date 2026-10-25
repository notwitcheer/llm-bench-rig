#!/bin/bash
# 68-ornith122b.sh: idle-queue item, STAGED 2026-10-25 by mercury (subagent, GPU thread). sorts after 67-sub10, before 69-hermes-kv-v3 (stays last).
# EV: private (projected post angle and reach bracket live in the ops vault status note)
# serves: content (model verdict)
# Ornith-1.5-122B UD-IQ3_XXS (unsloth GGUF, 49,515,667,712 B = 46.10 GiB, qwen35moe 122B-A10B, 256 experts / 8 used, 49 blocks incl. 1 nextn)
# on ONE RTX 5090 (32 GB) + 59 GB RAM with MoE expert offload (--n-cpu-moe), against the Ornith 1.5 35B Q4_K_M row (same arch family).
# build: the rig's mainline llama.cpp-nl b10371 (5d16e81dd), the SAME build the 35B depth rows ran on and config.yaml's server_bin; carries qwen35moe
#   (libllama strings + CPU-only dry load of this very gguf 2026-10-25: "qwen35moe 122B.A10B IQ3_XXS 3.0625 bpw 46.10 GiB 124.64 B"). no override, config.yaml untouched.
# legs, cheapest content first (every leg resumable on exact-path artifacts, hard stop = end of the heures creuses block, exit 3 while legs remain):
#   sweep          open-ended --n-cpu-moe ladder 34 -> 18 step 2 (llama-bench p512/n128, VRAM sampled), stops at first load failure / tg drop / VRAM over cap;
#                  opt = fastest point whose peak <= OPT_CAP_MIB. a page-cache/RAM check line per point.
#   speed          at opt: pp128..16384 + tg128 (card-fit VRAM peak), then depth 0/8k/32k TWICE (rep a/b, page-cache rule) -> speed.json
#   humaneval_plus_on, gpqa_off, board5_off (mmlu/arc/hellaswag/humaneval/gsm8k, config sample 0.5, think-off, ctx 8192), gpqa_on
#                  = the cells the Ornith 1.5 35B row has, same regime: think-on max_tokens 16384 ctx 24576 -> results/<slug>-thinkon/ (35B gpqa-on 81.82);
#                  think-off class-default max_tokens ctx 8192 -> results/<slug>/ (35B gpqa-off 52.02; board 82.24/94.11/91.0/87.8/91.66).
#   TAIL (only when ~/ornith122b-tail.enable exists): mbpp_plus_on math500_on ifeval_on, ma_aime_2026 + ma_hmmt_feb_2026 (think-on 32768 / ctx 40960, exact veto),
#                  then the four think-off second-tier cells.  not in the default plan: ~21 h of card at estimated pace.
# NOT here: toolcall-v1. its harness is pinned to the Qwen3.8 official template + parser; this gguf embeds a different template (sha256 9dd2fbd2... vs official c3cf9e34...)
#   and the 35B has no toolcall row to compare, so a leg needs a template pin + roundtrip first (CPU work, no card).
# greedy for every row (card sampling is temp 0.6 / top_k 20 / top_p 0.95: flag it in the readout).
set -uo pipefail
NL_TAG=ornith122b
MANIFEST=/home/witcheer/ornith122b-status.json
RUNLOG=/home/witcheer/ornith122b-run.log
export MANIFEST
. /home/witcheer/night-lib.sh
RIG=/home/witcheer/benchmark-rig
PY=$RIG/.venv/bin/python
GGUFPY=/home/witcheer/llama.cpp/gguf-py
BENCH=/home/witcheer/llama.cpp-nl/build/bin/llama-bench
SRV=/home/witcheer/llama.cpp-nl/build/bin/llama-server
BUILD_PIN=5d16e81
MDIR=/home/witcheer/models/ornith-15-122b
MODEL=$MDIR/Ornith-1.5-122B-UD-IQ3_XXS.gguf
GGUF_BYTES=49515667712
SLUG=ornith-1-5-122b-ud-iq3-xxs
OUT=$RIG/results/$SLUG; ON=$RIG/results/$SLUG-thinkon; MA=$RIG/results/$SLUG-matharena-32k; SWEEP=$OUT/offload-sweep
SWEEP_POINTS="34 32 30 28 26 24 22 20 18"   # --n-cpu-moe, high (safe) to low; open-ended: stops at first failure / tg drop / VRAM cap
SWEEP_STOP_MIB=29800                        # a point above this ends the ladder (card is 32,607 MiB; ctx 24-40k server adds ~1.5 GiB over the bench point)
OPT_CAP_MIB=28700                           # opt = fastest point at or under this peak (leaves room for the ctx 24576/40960 server, slots, depth 32k)
CORE_CELLS="humaneval_plus:on gpqa:off board5:off gpqa:on"
TAIL_CELLS="mbpp_plus:on math500:on ifeval:on ma_aime_2026:on ma_hmmt_feb_2026:on ifeval:off math500:off humaneval_plus:off mbpp_plus:off"
TAIL_FLAG=/home/witcheer/ornith122b-tail.enable
CELLS="$CORE_CELLS"; [ -f "$TAIL_FLAG" ] && CELLS="$CORE_CELLS $TAIL_CELLS"

cid() { echo "${1%%:*}_${1##*:}"; }   # cell id for the manifest: humaneval_plus_on
art_of() { # <task:think>
  local t=${1%%:*} m=${1##*:}
  case "$t:$m" in
    board5:off) echo "$OUT/quality.json" ;;
    gpqa:off) echo "$OUT/gpqa.json" ;;
    gpqa:on) echo "$ON/gpqa.json" ;;
    ma_*:on) echo "$MA/${t}_detail.json" ;;
    *:on) echo "$ON/${t}_detail.json" ;;
    *:off) echo "$OUT/${t}_detail.json" ;;
  esac
}

nl_say "PLAN phase (DRYRUN=${DRYRUN:-0})"
nl_check_path "$PY" "$BENCH" "$SRV" "$GGUFPY/gguf/__init__.py" "$RIG/lib/quality.py" "$RIG/lib/evals/gpqa.py" "$RIG/lib/evals/ifeval.py" "$RIG/lib/evals/math500.py" "$RIG/lib/evals/evalplus.py" "$RIG/lib/evals/matharena.py"
if [ ! -f "$MODEL" ]; then nl_say "  FAIL: gguf MISSING: $MODEL"; exit 2; fi
got=$(stat -c %s "$MODEL")
[ "$got" = "$GGUF_BYTES" ] && nl_ok "gguf $got B (byte-matched to HF unsloth/Ornith-1.5-122B-GGUF)" || { nl_say "  FAIL: gguf $got B, want $GGUF_BYTES B"; exit 2; }
if ls "$MDIR"/.cache/huggingface/download/*.incomplete >/dev/null 2>&1; then nl_say "  FAIL: hf .incomplete present, download not finished"; exit 2; fi
case "$(CUDA_VISIBLE_DEVICES= $SRV --version 2>&1)" in *"$BUILD_PIN"*) nl_ok "llama-server build $BUILD_PIN (config.yaml server_bin)";; *) nl_say "  FAIL: llama-server build changed (want $BUILD_PIN)"; exit 2;; esac
# llama-bench has no --version: assert the worktree HEAD it was built from + the b10371 shared libs it loads
hd=$(git -C /home/witcheer/llama.cpp-nl rev-parse HEAD 2>/dev/null || echo none)
if [ "${hd#"$BUILD_PIN"}" != "$hd" ] && ls /home/witcheer/llama.cpp-nl/build/bin/libllama.so.0.0.10371 >/dev/null 2>&1; then nl_ok "llama-bench worktree HEAD $hd, libllama 0.0.10371 (b10371)"; else nl_say "  FAIL: llama.cpp-nl HEAD '$hd' / libllama changed (want $BUILD_PIN, b10371)"; exit 2; fi
n=$(strings /home/witcheer/llama.cpp-nl/build/bin/libllama.so 2>/dev/null | grep -ciE "^qwen35moe$"); [ "${n:-0}" -gt 0 ] && nl_ok "libllama carries arch qwen35moe ($n hit)" || { nl_say "  FAIL: arch qwen35moe not in libllama.so"; exit 2; }
HELP=$(CUDA_VISIBLE_DEVICES= $BENCH --help 2>&1 || true)
for want in "--n-cpu-moe" "--n-depth" "--flash-attn"; do case "$HELP" in *"$want"*) ;; *) nl_fail "llama-bench lacks $want";; esac; done
SHELP=$(CUDA_VISIBLE_DEVICES= $SRV --help 2>&1 || true)
case "$SHELP" in *"--n-cpu-moe"*) nl_ok "llama-bench + llama-server list --n-cpu-moe / --n-depth / --flash-attn";; *) nl_fail "llama-server lacks --n-cpu-moe";; esac
# gguf header read, CPU only (no model load): arch, layers, experts, per-layer expert bytes -> predicted weights-on-GPU / experts-in-RAM per sweep point
PRED=$(nice -n 10 timeout 180 env CUDA_VISIBLE_DEVICES= $PY - "$MODEL" "$GGUFPY" "$SWEEP_POINTS" <<'PYEOF' 2>&1
import sys, collections
sys.path.insert(0, sys.argv[2])
from gguf import GGUFReader
r = GGUFReader(sys.argv[1]); f = r.fields
g = lambda k: f[k].contents()
arch = g("general.architecture"); nl = g(arch + ".block_count"); ne = g(arch + ".expert_count"); nu = g(arch + ".expert_used_count"); nx = g(arch + ".nextn_predict_layers")
assert arch == "qwen35moe" and nl == 49 and ne == 256 and nu == 8 and nx == 1, (arch, nl, ne, nu, nx)
G = 2 ** 30; lay = collections.defaultdict(int); non = 0
for t in r.tensors:
    if "_exps" in t.name: lay[int(t.name.split(".")[1])] += int(t.n_bytes)
    else: non += int(t.n_bytes)
assert len(lay) == nl, len(lay)
print(f"gguf: arch {arch}, {nl} blocks (1 nextn), {ne} experts / {nu} used, expert bytes {sum(lay.values())/G:.2f} GiB (blk48 nextn {lay[48]/G:.2f}), non-expert {non/G:.2f} GiB, {len(r.tensors)} tensors")
for n in sys.argv[3].split():
    n = int(n); ram = sum(lay[i] for i in range(n)); gpu = non + sum(lay[i] for i in range(n, nl))
    print(f"  n_cpu_moe={n}: experts in RAM {ram/G:.1f} GiB, weights on GPU {gpu/G:.1f} GiB (weights only; KV/compute/context add ~2-4 GiB)")
PYEOF
)
if [ $? -eq 0 ] && echo "$PRED" | grep -q "^gguf: arch qwen35moe"; then echo "$PRED" | while IFS= read -r l; do nl_ok "$l"; done; else nl_fail "gguf header check FAILED: $(echo "$PRED" | tail -2 | tr '\n' ' ')"; fi
free_g=$(free -g | awk '/^Mem:/{print $2}'); [ "$free_g" -ge 55 ] && nl_ok "RAM ${free_g}G total (file 46.1 GiB fits the page cache whole)" || nl_fail "RAM ${free_g}G < 55"
nl_check_disk 30
nl_say "  gpu untouched in PLAN (every probe above ran with the card hidden)"
$PY -c "from datasets import load_dataset; d=load_dataset('Idavidrein/gpqa','gpqa_diamond',split='train'); assert len(d)==198" >/dev/null 2>&1 && nl_ok "gpqa_diamond loads (198)" || nl_fail "gpqa dataset probe FAILED"
$PY -c "
from datasets import load_dataset
assert len(load_dataset('evalplus/humanevalplus', split='test')) == 164
" >/dev/null 2>&1 && nl_ok "humaneval+ dataset loads (164)" || nl_fail "humaneval+ dataset probe FAILED"
if [ -f "$TAIL_FLAG" ]; then
  $PY -c "
from datasets import load_dataset
for name, cfg, split, n in [('google/IFEval', None, 'train', 541), ('HuggingFaceH4/MATH-500', None, 'test', 500), ('evalplus/mbppplus', None, 'test', 378), ('MathArena/aime_2026', None, 'train', 30), ('MathArena/hmmt_feb_2026', None, 'train', 33)]:
    assert len(load_dataset(name, cfg, split=split)) == n, name
" >/dev/null 2>&1 && nl_ok "tail datasets load (ifeval, math500, mbpp+, 2 matharena)" || nl_fail "tail dataset probe FAILED"
  (cd $RIG && $PY -c "from lib.qwen_math_grader import grade; assert grade(r'\\frac{1}{2}','0.5')") >/dev/null 2>&1 && nl_ok "math grader imports" || nl_fail "math grader broken"
  $PY -c "import sys; sys.path.insert(0,'$RIG'); from latex2sympy2_extended import latex2sympy; from lib.evals.matharena import grade_with_timeout as g; assert g('277','277', veto=True)[0]" >/dev/null 2>&1 && nl_ok "exact-number veto" || nl_fail "exact-number veto check FAILED"
  nl_say "  TAIL ENABLED ($TAIL_FLAG present)"
else nl_say "  tail legs off (touch $TAIL_FLAG to add them)"; fi

# plan: one RUN/SKIP line per leg, literal artifact paths
PH="sweep speed"; RUNS=0; PRESENT=""
[ -f "$SWEEP/opt.txt" ] && { nl_say "  sweep: present ($SWEEP/opt.txt = $(cat "$SWEEP/opt.txt"))"; PRESENT="$PRESENT sweep"; } || { nl_say "  sweep: RUN"; RUNS=$((RUNS+1)); }
[ -f "$OUT/speed.json" ] && { nl_say "  speed: present"; PRESENT="$PRESENT speed"; } || { nl_say "  speed: RUN"; RUNS=$((RUNS+1)); }
for c in $CELLS; do a=$(art_of "$c"); PH="$PH $(cid "$c")"
  [ -f "$a" ] && { nl_say "  $(cid "$c"): present"; PRESENT="$PRESENT $(cid "$c")"; } || { nl_say "  $(cid "$c"): RUN ($a)"; RUNS=$((RUNS+1)); }; done
DL=$(nl_deadline_epoch); nl_say "  planned RUN: $RUNS (hard stop $(date -d "@$DL" '+%a %H:%M') local, $(( (DL-$(date +%s))/60 )) min away)"
[ "$RUNS" -gt 0 ] || { nl_say "nothing to run, item complete"; exit 0; }
mkdir -p "$OUT" "$SWEEP" "$ON"
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init $PH; for p in $PRESENT; do nl_mark "$p" skipped pre-existing; done; fi
# harness self-test, no server, no GPU: every eval class constructs with the exact arguments the legs use
NL_SELFTEST=1 CUDA_VISIBLE_DEVICES= PYTHONPATH="$RIG" $PY - <<'PYEOF' >/dev/null 2>&1 && nl_ok "harness self-test: eval classes construct (gpqa/ifeval/math500/evalplus/matharena)" || nl_fail "harness self-test FAILED (eval class signature changed)"
import json, tempfile
from pathlib import Path
import lib.config as _c
from lib.evals.base import LLMClient
from lib.evals.gpqa import GPQAEval
from lib.evals.ifeval import IFEvalEval
from lib.evals.math500 import Math500Eval
from lib.evals.evalplus import EvalPlusEval
from lib.evals.matharena import MathArenaEval
rd = Path(tempfile.mkdtemp())
c = LLMClient("http://127.0.0.1:1/v1", "x", think=True, timeout=5)
GPQAEval(client=c, results_dir=rd, max_tokens=16384); GPQAEval(client=c, results_dir=rd)
IFEvalEval(client=c, results_dir=rd, max_tokens=16384); Math500Eval(client=c, results_dir=rd, max_tokens=16384)
EvalPlusEval(client=c, variant="humaneval_plus", results_dir=rd, max_tokens=16384)
MathArenaEval(client=c, task="ma_aime_2026", results_dir=rd, max_tokens=32768, exact_veto=True)
from lib.quality import build_server_command
cmd = build_server_command("/m.gguf", port=8090, ngl=99, ctx_size=24576, n_cpu_moe=26)
assert "--n-cpu-moe" in cmd and "--jinja" in cmd and cmd[0].endswith("llama.cpp-nl/build/bin/llama-server"), cmd
PYEOF
nl_gate

# ---------------- GPU phases: drain Donald, EXIT trap restores ----------------
DEADLINE=$(nl_deadline_epoch)
left_s() { echo $(( DEADLINE - $(date +%s) )); }
tmo() { local l; l=$(left_s); [ "$l" -gt "$1" ] && l=$1; [ "$l" -lt 60 ] && l=60; echo "$l"; }   # timeout capped at the hard stop
SMI=""
stop_smi() { [ -n "$SMI" ] && { kill "$SMI" 2>/dev/null; wait "$SMI" 2>/dev/null; SMI=""; }; return 0; }
restore() { stop_smi; pkill -9 -x llama-server 2>/dev/null; pkill -9 -x llama-bench 2>/dev/null; sleep 5; nl_say "EXIT trap: restoring Donald"; systemctl is-active --quiet llama-server.service || sudo -n systemctl start llama-server.service || true; }
trap restore EXIT
nl_say "draining Donald"; sudo -n systemctl stop llama-server.service; sleep 3
for i in $(seq 1 120); do v=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '); [ "${v:-99999}" -lt 1100 ] && { nl_say "drain clear (${v}MiB)"; break; }; sleep 2; [ "$i" -eq 120 ] && { nl_say "FATAL drain timeout"; exit 1; }; done
cd "$RIG"; export PYTHONPATH="$RIG"

# ---------- leg: --n-cpu-moe sweep (open-ended, high N to low N) ----------
OPT_FILE=$SWEEP/opt.txt
if nl_skip_exact "$OPT_FILE"; then nl_say "sweep: SKIP (opt $(cat "$OPT_FILE"))"
elif [ "$(left_s)" -lt 1800 ]; then nl_say "sweep: window low ($(left_s) s), resumable"; nl_mark sweep failed "window exhausted (resumable)"
else
  t0=$(date +%s); nl_mark sweep running; nl_say "=== sweep start ($(( $(left_s) / 60 )) min left) points: $SWEEP_POINTS, vram stop ${SWEEP_STOP_MIB} MiB, opt cap ${OPT_CAP_MIB} MiB ==="
  prev_tg=0
  for N in $SWEEP_POINTS; do
    [ "$(left_s)" -lt 900 ] && { nl_say "  sweep: window low, stopping ladder"; break; }
    if [ ! -f "$SWEEP/bench_$N.json" ]; then
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$SWEEP/vram_$N.txt" 2>/dev/null & SMI=$!
      lim=$(left_s); [ "$lim" -gt 1200 ] && lim=1200
      if timeout "$lim" "$BENCH" -m "$MODEL" -ngl 99 --n-cpu-moe "$N" -fa 1 -p 512 -n 128 -r 2 -o json > "$SWEEP/bench_$N.json" 2> "$SWEEP/bench_$N.err"; then :
      else nl_say "  n_cpu_moe=$N FAILED (load or run): $(grep -iE 'out of memory|failed|error' "$SWEEP/bench_$N.err" | head -1 | cut -c1-120)"; rm -f "$SWEEP/bench_$N.json"; stop_smi; sleep 5; break; fi
      stop_smi; sleep 5
    fi
    tg=$($PY -c "import json,sys; r=[x for x in json.load(open(sys.argv[1])) if x['n_gen']]; print(round(r[0]['avg_ts'],2))" "$SWEEP/bench_$N.json" 2>/dev/null || echo 0)
    pp=$($PY -c "import json,sys; r=[x for x in json.load(open(sys.argv[1])) if x['n_prompt']]; print(round(r[0]['avg_ts'],1))" "$SWEEP/bench_$N.json" 2>/dev/null || echo 0)
    [ "$tg" = "0" ] && { nl_say "  n_cpu_moe=$N: unreadable bench json, dropped"; rm -f "$SWEEP/bench_$N.json"; break; }
    pk=$(sort -n "$SWEEP/vram_$N.txt" 2>/dev/null | tail -1); pk=${pk:-0}
    nl_say "  n_cpu_moe=$N tg128 $tg tok/s, pp512 $pp, vram peak $pk MiB, RAM free $(free -g | awk '/^Mem:/{print $4}')G avail $(free -g | awk '/^Mem:/{print $7}')G"
    [ "$pk" -gt "$SWEEP_STOP_MIB" ] && { nl_say "  vram peak over ${SWEEP_STOP_MIB} MiB, ladder ends here"; break; }
    if awk -v a="$tg" -v b="$prev_tg" 'BEGIN{exit !(b>0 && a < b*0.99)}'; then nl_say "  tg dropped vs previous point ($prev_tg -> $tg), ladder ends here"; break; fi
    prev_tg=$tg
  done
  $PY - "$SWEEP" "$OPT_CAP_MIB" <<'PYEOF' >> "$RUNLOG" 2>&1
import json, sys, glob, os
sw, cap = sys.argv[1], int(sys.argv[2]); rows = []
for f in glob.glob(os.path.join(sw, "bench_*.json")):
    n = int(os.path.basename(f)[6:-5])
    try:
        r = json.load(open(f)); tg = [x for x in r if x["n_gen"]][0]; pp = [x for x in r if x["n_prompt"]][0]
        v = open(os.path.join(sw, f"vram_{n}.txt")).read().split(); pk = max(int(x) for x in v) if v else None
        rows.append({"n_cpu_moe": n, "tg128": round(tg["avg_ts"], 2), "tg128_std": round(tg["stddev_ts"], 2), "pp512": round(pp["avg_ts"], 1), "vram_peak_mib": pk})
    except Exception as e: rows.append({"n_cpu_moe": n, "failed": str(e)[:80]})
rows.sort(key=lambda r: -r["n_cpu_moe"]); json.dump(rows, open(os.path.join(sw, "sweep.json"), "w"), indent=2)
ok = [r for r in rows if "tg128" in r and r["vram_peak_mib"] and r["vram_peak_mib"] <= cap]
if not ok: print("sweep: no point under the cap"); sys.exit(1)
best = max(ok, key=lambda r: r["tg128"]); open(os.path.join(sw, "opt.txt"), "w").write(str(best["n_cpu_moe"])); print("sweep:", json.dumps(rows), "opt", best)
PYEOF
  nl_judge_artifact sweep $? "$OPT_FILE" "$t0" "offload sweep" && nl_say "=== sweep done: opt n_cpu_moe=$(cat "$OPT_FILE") ===" || { nl_say "=== sweep FAILED (no point loaded under the cap; ladder resumable) ==="; exit 1; }
fi
OPT=$(cat "$OPT_FILE")

# ---------- leg: speed at opt (pp ladder + tg, then depth twice) ----------
if nl_skip_exact "$OUT/speed.json"; then nl_say "speed: SKIP"
elif [ "$(left_s)" -lt 1800 ]; then nl_say "speed: window low, resumable"; nl_mark speed failed "window exhausted (resumable)"
else
  t0=$(date +%s); nl_mark speed running; nl_say "=== speed start ($(( $(left_s) / 60 )) min left) n_cpu_moe=$OPT ==="
  [ -f "$OUT/bench.pp.json" ] || { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$OUT/vram_speed.txt" 2>/dev/null & SMI=$!
    timeout "$(tmo 1800)" "$BENCH" -m "$MODEL" -ngl 99 --n-cpu-moe "$OPT" -fa 1 -p 128,512,2048,4096,8192,16384 -n 128 -r 2 -o json > "$OUT/bench.pp.json" 2>> "$RUNLOG" || rm -f "$OUT/bench.pp.json"; stop_smi; }
  for rep in a b; do
    [ -f "$OUT/bench.depth.$rep.json" ] && continue
    [ "$(left_s)" -lt 1200 ] && { nl_say "  speed: window low before depth rep $rep"; break; }
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$OUT/vram_depth_$rep.txt" 2>/dev/null & SMI=$!
    timeout "$(tmo 2400)" "$BENCH" -m "$MODEL" -ngl 99 --n-cpu-moe "$OPT" -fa 1 -p 512 -n 128 -d 0,8192,32768 -r 2 -o json > "$OUT/bench.depth.$rep.json" 2>> "$RUNLOG" || rm -f "$OUT/bench.depth.$rep.json"
    stop_smi
  done
  $PY - "$OUT" "$OPT" <<'PYEOF' >> "$RUNLOG" 2>&1
import json, sys, os
from pathlib import Path
out = Path(sys.argv[1]); opt = int(sys.argv[2])
def summ(rows):
    s = {}
    for r in rows:
        test = f"pp{r['n_prompt']}" if r["n_prompt"] else f"tg{r['n_gen']}"; d = r.get("n_depth", 0)
        if d: test += f"@d{d}"
        s[test] = {"tokens_per_sec": round(r["avg_ts"], 2), "stddev": round(r["stddev_ts"], 2)}
    return s
def peak(*names):
    v = []
    for n in names:
        p = out / n
        if p.exists(): v += [int(x) for x in p.read_text().split() if x.isdigit()]
    return max(v) if v else None
pp = summ(json.load(open(out / "bench.pp.json"))); a = summ(json.load(open(out / "bench.depth.a.json")))
sp = dict(pp)
for k, v in a.items():
    if "@d" in k: sp[k] = v
sp["model_size_gib"] = 46.10; sp["params"] = "124.64 B (10 B active, qwen35moe 256 experts / 8 used)"; sp["backend"] = "CUDA"
sp["vram_peak_mib"] = peak("vram_speed.txt"); sp["vram_peak_depth_mib"] = peak("vram_depth_a.txt", "vram_depth_b.txt")
sp["engine"] = "llama.cpp"; sp["n_cpu_moe"] = opt; sp["offload"] = "experts of the first n_cpu_moe layers in system RAM"
sp["build"] = "b10371 5d16e81dd (llama.cpp-nl)"; sp["ram_gb"] = 59
if (out / "bench.depth.b.json").exists():
    b = summ(json.load(open(out / "bench.depth.b.json"))); sp["depth_rep_b"] = {k: v for k, v in b.items() if "@d" in k or k in ("tg128", "pp512")}
    sp["depth_rep_spread_pct"] = {k: round(100 * abs(a[k]["tokens_per_sec"] - b[k]["tokens_per_sec"]) / a[k]["tokens_per_sec"], 1) for k in a if k in b}
json.dump(sp, open(out / "speed.json", "w"), indent=2); print("speed:", json.dumps(sp))
PYEOF
  nl_judge_artifact speed $? "$OUT/speed.json" "$t0" "$SLUG offload speed+depth" && nl_say "=== speed done: $($PY -c "import json;s=json.load(open('$OUT/speed.json'));print('tg128',s.get('tg128',{}).get('tokens_per_sec'),'pp512',s.get('pp512',{}).get('tokens_per_sec'),'tg@32k',s.get('tg128@d32768',{}).get('tokens_per_sec'),'vram_peak',s.get('vram_peak_mib'),'depth_spread',s.get('depth_rep_spread_pct'))" 2>/dev/null) ===" || nl_say "=== speed FAILED ==="
  [ -f "$OUT/speed.json" ] && { CUDA_VISIBLE_DEVICES= $SRV --version > "$OUT/llama-server-version.txt" 2>&1; { echo "model: $MODEL $(stat -c %s "$MODEL") bytes"; echo "n_cpu_moe opt: $OPT"; echo "build: $BUILD_PIN"; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; free -g | head -2; } > "$OUT/env.txt" 2>&1 || true; }
fi

# ---------- cells: server legs, same regimes as the Ornith 1.5 35B row ----------
run_cell() { # <task:think> <opt>  -> rc
  local cell=$1 o=$2 t=${1%%:*} m=${1##*:} left; left=$(( DEADLINE - $(date +%s) - 120 ))
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -l 1 > "$OUT/vram_$(cid "$cell").txt" 2>/dev/null & SMI=$!
  timeout "$left" $PY -u - "$MODEL" "$SLUG" "$t" "$m" "$o" >> "$RUNLOG" 2>&1 <<'PYEOF'
import sys, json, time, os
from pathlib import Path
import lib.quality as q
from lib.evals.base import LLMClient
path, slug, t, mode, opt = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
think = mode == "on"
base = Path("results")
if t.startswith("ma_"): think_max, ctx, rd, out = 32768, 40960, base / f"{slug}-matharena-32k", f"{t}_detail.json"
elif think: think_max, ctx, rd, out = 16384, 24576, base / f"{slug}-thinkon", ("gpqa.json" if t == "gpqa" else f"{t}_detail.json")
else: think_max, ctx, rd, out = 0, 8192, base / slug, ("quality.json" if t == "board5" else "gpqa.json" if t == "gpqa" else f"{t}_detail.json")
rd.mkdir(parents=True, exist_ok=True)
off = f"n_cpu_moe {opt}, llama.cpp b10371 5d16e81dd, 59 GB RAM"
if t == "board5":
    from lib.config import load_config
    _c = load_config(); _c.setdefault("quality", {})["mc_gate_tokens"] = 400   # gate stays armed for true degeneracy, tolerates the odd essay (2026-09-08 offload precedent)
    print(f"[{slug}] board5 server: {' '.join(q.build_server_command(path, port=8090, ngl=99, ctx_size=ctx, n_cpu_moe=opt))}", flush=True)
    t0 = time.time()
    r = q.run_quality_bench(path, engine="llama.cpp", results_dir=rd, offload={"ctx_size": ctx, "n_cpu_moe": opt})
    (rd / "quality.regime.txt").write_text(f"think-off board (config sample 0.5 seed 42, mc_gate_tokens 400), ctx {ctx}, greedy, {off}\n")   # quality.json stays a pure {task: {score}} dict
    (rd / out).write_text(json.dumps(r, indent=2))
    print(f"[{slug}] board5:{mode}: {json.dumps({k: (v.get('score') if isinstance(v, dict) else v) for k, v in r.items()})} in {time.time()-t0:.0f}s", flush=True)
    sys.exit(0)
print(f"[{slug}] {t}:{mode} server: {' '.join(q.build_server_command(path, port=8090, ngl=99, ctx_size=ctx, n_cpu_moe=opt))}", flush=True)
proc = q.start_llama_server(path, ctx_size=ctx, n_cpu_moe=opt)
try:
    with LLMClient("http://127.0.0.1:8090/v1", Path(path).stem, think=think, timeout=3600) as client:
        t0 = time.time(); kw = {"max_tokens": think_max} if think_max else {}
        if t == "gpqa":
            from lib.evals.gpqa import GPQAEval; ev = GPQAEval(client=client, results_dir=rd, **kw)
        elif t == "ifeval":
            from lib.evals.ifeval import IFEvalEval; ev = IFEvalEval(client=client, results_dir=rd, **kw)
        elif t == "math500":
            from lib.evals.math500 import Math500Eval; ev = Math500Eval(client=client, results_dir=rd, **kw)
        elif t in ("humaneval_plus", "mbpp_plus"):
            from lib.evals.evalplus import EvalPlusEval; ev = EvalPlusEval(client=client, variant=t, results_dir=rd, **kw)
        else:
            from lib.evals.matharena import MathArenaEval; ev = MathArenaEval(client=client, task=t, results_dir=rd, exact_veto=True, **kw)
        r = ev.evaluate()
        mt = think_max or getattr(ev, "max_tokens", 2048)
        r["regime"] = (f"think-on, max_tokens {think_max}, ctx {ctx}, zero-shot, greedy, {off}" if think else f"think-off, max_tokens {mt} (class default), ctx {ctx}, zero-shot, greedy, {off}")
        r["bench"] = "Ornith-1.5-122B UD-IQ3_XXS with MoE expert offload vs the Ornith 1.5 35B Q4_K_M row (same cells, same eval classes, greedy; card sampling is temp 0.6 / top_k 20 / top_p 0.95)"
        r["model_file"] = "unsloth/Ornith-1.5-122B-GGUF Ornith-1.5-122B-UD-IQ3_XXS.gguf (49,515,667,712 B)"
        (rd / out).write_text(json.dumps(r, indent=2))
        print(f"[{slug}] {t}:{mode}: {json.dumps({k: r.get(k) for k in ('score','metric','correct','total','parse_failures','tokens_per_correct','capped_rate')})} in {time.time()-t0:.0f}s", flush=True)
finally:
    q.stop_llama_server(proc)
PYEOF
  local rc=$?; pkill -9 -x llama-server 2>/dev/null; sleep 3; stop_smi
  cp -f ~/bench-server.log "$OUT/server-$(cid "$cell").log" 2>/dev/null || true
  echo "$o" > "$OUT/n_cpu_moe_$(cid "$cell").txt"
  grep -qiE "watchdog|launch timeout|illegal|CUDA error" "$OUT/server-$(cid "$cell").log" 2>/dev/null && nl_say "  RED: CUDA fault text in server-$(cid "$cell").log"
  return $rc
}
for c in $CELLS; do
  art=$(art_of "$c"); id=$(cid "$c")
  if nl_skip_exact "$art"; then continue; fi
  left=$(( DEADLINE - $(date +%s) ))
  [ "$left" -gt 1200 ] || { nl_say "window exhausted before $id ($left s left), resumable"; nl_mark "$id" failed "window exhausted (resumable)"; continue; }
  t0=$(date +%s); nl_mark "$id" running; nl_say "=== $id (n_cpu_moe $OPT, budget $((left/60)) min) ==="
  rc=0
  for o in "$OPT" $((OPT+2)); do
    [ "$o" -le 48 ] || break
    run_cell "$c" "$o"; rc=$?
    [ "$rc" = 0 ] && break
    [ "$rc" = 124 ] && break
    if tail -300 ~/bench-server.log 2>/dev/null | grep -qiE "out of memory|cudaMalloc|failed to allocate|CUDA error"; then nl_say "  $id: server OOM at n_cpu_moe=$o, retrying once two layers further out"; sleep 5; continue; fi
    break
  done
  vp=$(sort -n "$OUT/vram_$id.txt" 2>/dev/null | tail -1)
  if nl_judge_artifact "$id" $rc "$art" "$t0" "$(( ($(date +%s)-t0)/60 )) min, vram peak ${vp:-?} MiB"; then nl_say "  $(grep -a "\[$SLUG\] $c:" "$RUNLOG" | tail -1 | cut -c1-240) vram peak ${vp:-?} MiB"; else nl_say "  $id: FAILED rc=$rc (124 at the hard stop = resumable)"; fi
done
restore; trap - EXIT
ok=0; for _ in $(seq 1 60); do curl -s -m 3 localhost:8090/health | grep -qi ok && { ok=1; break; }; sleep 5; done
[ "$ok" = "1" ] && nl_say "Donald healthy at end" || { nl_say "DONALD NOT HEALTHY, restoring"; nl_donald_restore; }
[ -f "$OPT_FILE" ] && [ -f "$OUT/speed.json" ] || { nl_say "legs remain, resumable"; exit 3; }
for c in $CELLS; do [ -f "$(art_of "$c")" ] || { nl_say "legs remain, resumable"; exit 3; }; done
nl_say "all legs done"; exit 0
