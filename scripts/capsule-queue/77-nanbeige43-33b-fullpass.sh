#!/bin/bash
# 77-nanbeige43-33b-fullpass.sh: idle-queue item, STAGED 2026-10-30 by mercury (subagent, GPU thread; operator picked Nanbeige4.3-33B as a weekend fill). Sorts after 73/74, never before the Sat VRAM-tier readout margin.
# EV: arguable claim: the card says Nanbeige4.3-33B beats Qwen3.8-27B (Terminal-Bench 2.1 80.9 vs 73.0, DeepSWE 65.5 vs 42.5, GPQA Diamond 89.5 vs 89.2, AIME 2026 95.2 vs 94.5); we hold the deepest Qwen3.8-27B rows of any harness we run
#   (toolcall-v1 S5 81/100, agenttest v2 30/30 on Q6_K), so the challenger-vs-champion check on Hermes-shaped tool calls is ours. Gate rule 1 (which-file on a brand-pull model: the Qwen which-file post did 11,302 views / 207 bookmarks) plus the
#   context-ceiling finding "which file fits a 32 GB card at what context" (KV 208 KiB/token, 3.2x Qwen3.8-27B's: Q6_K is predicted not to hold Hermes' 64k at all, UNVERIFIED until the fit leg). Line 1 follows rule A below, never "first".
#   Comparables [GATE, IMP]: Ornith 122B verdict 3,635; Muse Glimmer 3,901; which-file Qwen 11,302. Nanbeige has no brand pull on our account (gate rule 8 floor 1-1.7k): the Qwen comparison carries it. Honest bracket 2-6k, a table alone sits at the floor.
#   Mojo (rule 9a): the arguable claim (card vs our own harness, one 5090) + a tokenizer trap in the widely downloaded GGUF (91k downloads in 5 days), measured and fixed here. Picture: the ceiling chart (context vs KV type per file) and the S5 paired bars.
# serves: content (a challenger-vs-champion verdict on Hermes-shaped tool calls + a which-file / context-ceiling finding) + hardening (a second template and tokenizer through toolcall-v1, a tokenizer trap found and fixed on a GGUF)
#
# WHAT: Nanbeige4.3-33B (Nanbeige, 2026-10-24, apache-2.0, LlamaForCausalLM dense, 52 layers, hidden 7168, 56 heads / 8 KV heads, head_dim 128, vocab 166,912, KV 208 KiB/token f16) in two bartowski GGUFs on ONE RTX 5090,
#   against the Qwen3.8-27B rows we already hold on the same harnesses. The card claims it beats Qwen3.8-27B at the highest thinking effort (Terminal-Bench 2.1 80.9 vs 73.0, DeepSWE 65.5 vs 42.5,
#   SWE-Bench Pro 60.3 vs 61.7, Toolathlon-Verified 70.4 vs 68.5, Tau3 48.8 vs 48.5, GPQA Diamond 89.5 vs 89.2, AIME 2026 95.2 vs 94.5; https://huggingface.co/Nanbeige/Nanbeige4.3-33B, fetched 2026-10-30 ~13:50 CET).
#
# FILES (HF bartowski/Nanbeige_Nanbeige4.3-33B-GGUF @ 9beb68421fdf, byte + sha256 verified 2026-10-30, stat stamp ~/nanbeige43/gguf.stat is re-checked at every PLAN):
#   Q4_K_M 20,788,825,216 B sha256 2cef79855ca47ee6571e166db32a07089e34f3cf78e0397b262d124ddb814c37 | Q6_K 29,016,857,728 B sha256 b11c9f21e69d896a1c20b12f88174f069e0e15623140733a49ff8e47c5541fbc
#   IQ3_XXS not downloaded: a 33B cannot hold Hermes' 64k context on a 16 GB tier whatever the weights (KV alone is 6.9 GiB at q8_0), so it would only measure a NOFIT.
#
# TWO TRAPS FOUND AT STAGING (CPU, 2026-10-30, evidence in the vault note), both handled here:
#   (1) the GGUF has add_bos_token=true with BOS = <|im_start|>: a raw /completion text prompt gets a second one -> toolcall-v1 sends token-id arrays (driver --prompt-tokens, ids from the HF tokenizer).
#   (2) the GGUF has tokenizer.ggml.add_space_prefix=true, so llama.cpp's SPM path inserts a "▁" after every special token (<|im_start|>▁user, <|im_end|>▁\n); the HF tokenizer does not, and the model was trained on HF ids.
#       server /tokenize != HF ids on 1,998/1,998 frozen prompts stock; with `--override-kv tokenizer.ggml.add_space_prefix=bool:false` it is 0/1,998 on b10371 AND on the bolt build. EVERY server in this item
#       carries that override (a per-server guard asserts it) and says so in the regime string. The stock-tokenizer cost is its own optional leg (touch ~/nanbeige43-extra.enable, see EXTRA).
#
# BUILDS: board legs (fit, speed, board, toolcall, gpqa, aime) = ~/llama.cpp-nl b10371 (5d16e81dd), config.yaml server_bin, the build the Qwen3.8-27B board/toolcall/gpqa/aime rows ran on;
#   agenttest leg = ~/llama.cpp-bolt build 11668 (cd0a5b6a1), the build the 73 rows ran on. Hermes pin as 73.
#
# LEGS, per file, in this order; ALL Q4_K_M legs before ANY Q6_K leg; every leg resumable on exact-path artifacts; exit 3 while legs remain; a leg is not started with fewer minutes left than MIN_MIN[leg]:
#   fit    context ceiling, f16 and q8_0 KV: ladder 8192..262144 (own llama-server per point, ~6k prefill, per-pid VRAM peak, --fit off), ceiling = largest ctx that loaded AND answered (nb_fit.py)
#   speed  llama-bench in the board regime (pp128..16384 + tg128, VRAM peak) with fit fallbacks, depth 0/8k/32k f16 AND q8_0 (nb_speed.py) -> results/<slug>/speed.json, the decode note per file
#   board5 mmlu/arc/hellaswag/humaneval/gsm8k, config sample 0.5, think-off, greedy (nb_cell.py -> lib.quality.run_quality_bench), ctx 8192
#   tc_st  toolcall-v1 st-on: S1..S4 single-turn (899 requests), frozen ids, think-on, max_tokens 16384, greedy, parser-2, planned ctx 24,576 (driver code-nb, token-id prompts)
#   tc_s5  toolcall-v1 s5-on: S5 multi_turn_base, 100 entries, think-on, planned ctx 40,960
#   agent  agenttest v2 row: the 30 frozen tasks, real Hermes (pin as 73), -c 65536, own rows file rows-nanbeige.json, own results dir ~/agenttest-v2/results/nb43; f16 KV first, q8kv row only where f16 is NOFIT
#   gpqa   GPQA-diamond think-on, 16,384 budget, greedy (second tier, capped_rate reported, never folded into q_avg), planned ctx 24,576
#   aime   MathArena AIME 2026 at 32,768 budget, think-on, greedy, exact-number veto, planned ctx 40,960 (cap reported; the card's 95.2 is at an unlimited budget: label the difference)
# Q6_K long poles (gpqa, aime, ~9.8 h) are dropped when ~/nanbeige43-q6-long.disable exists (the operator's cut switch); default is ALL legs.
# The KV type and context of every server leg come from the measured ceilings (nb_plan.py): f16 when the planned ctx fits f16, else q8_0 (labelled deviation from the Qwen rows), else q8_0 at the ceiling (flagged limited) when
#   that is at least the leg's minimum viable ctx, else the leg is recorded NOFIT (a result, not a retry). Every detail json carries kv + ctx in its regime string.
#
# ---------------------------------------------------------------- PRE-REGISTRATION (written before any Nanbeige output exists; nothing below is edited after a result)
# COMPARATORS (Qwen3.8-27B rows on the same harness, capsule paths; Q4_K_M vs Q4_K_M-class and Q6_K vs Q6_K where they exist):
#   board + speed : ~/benchmark-rig/results/qwen3-8-27b-q4-k-m/{quality,speed}.json (plain Q4_K_M, 15.93 GiB, tg128 78.94, q_avg 93.17)  |  .../qwen3-8-27b-q6-k/{quality,speed,bench.depth}.json (Q6_K, tg128 63.08, q_avg 93.66)
#   toolcall-v1   : ~/benchmark-rig/results/toolcall-v1/qwen3-8-27b-q6-k-{st-on,s5-on}/ (Q6_K: single-turn 799/906, S5 81/100) and .../qwen3-8-27b-ud-iq3-xxs-{st-on,s5-on}/ (815/906, S5 80/100). NO Qwen Q4_K_M toolcall row exists:
#                   Nanbeige Q4_K_M is compared with the Qwen Q6_K (a smaller challenger file against the champion's best file, the stricter test for Nanbeige) and said so.
#   agenttest v2  : ~/agenttest-v2/results/c1/qwen38-27b-ud-q4km (UD-Q4_K_M, the Q4_K_M-class row) and .../qwen38-27b-q6k (Q6_K) and .../qwen38-27b-ud-iq3xxs, same 30 tasks, same Hermes pin, bolt cd0a5b6a1
#   GPQA think-on : ~/benchmark-rig/results/qwen3-8-27b-q6-k-thinkon-rerun/gpqa.json (79.29, 157/198, greedy, 16k; capped_rate from its gpqa_tokens.jsonl sidecar) and qwen3-8-27b-ud-iq3-xxs-thinkon/gpqa.json (78.79). No Qwen Q4_K_M think-on row exists.
#   AIME 2026 32k : ~/benchmark-rig/results/qwen3-8-27b-q6-k-matharena-32k/ma_aime_2026_detail.json (83.33, 25/30, capped 0.2333) and qwen3-8-27b-ud-iq3-xxs-matharena-32k (86.67, capped 0.20); K=8 sampled Q6_K 85.83 in ...-s8.
# PAIRED TESTS between Nanbeige and Qwen THEMSELVES (never each vs a reference): exact two-sided McNemar per item, written by nb_readout.py on: toolcall single-turn (899 unique ids), S5 (100 ids), agenttest (30 tasks),
#   GPQA think-on (198), AIME (30); plus Nanbeige Q4_K_M vs Nanbeige Q6_K on S5 (the which-file pair). (The skill's scripts/hrl-paired-mcnemar.py reads harness-RL pair files, so nb_readout.py carries the same exact test.)
# TIE MARGINS (fixed now): board q_avg +-1.2 points; toolcall single-turn +-2.5 points; S5 and every item-level test: a win or loss needs exact p < 0.05, else TIE; GPQA +-3 points (198 items, one item 0.5);
#   AIME +-2 problems (30 items, one = 3.3 points); agenttest a lead under 3 of 30 tasks is a TIE (PREREG.md of agenttest v2, unchanged).
# LINE-1 RULE (A, applied by nb_readout.py to the S5 multi-turn pair, Nanbeige's best complete file vs Qwen3.8-27B Q6_K, the number a Hermes Agent user feels):
#   A-win  : "claims to beat Qwen3.8-27B; on Hermes Agent tool calls on one 5090 it wins N vs M"   when Nanbeige S5 > Qwen S5 and exact p < 0.05
#   A-tie  : "... it ties N vs M" (the reader gets: you cannot tell them apart on 100 multi-turn entries) otherwise
#   A-lose : "... it loses N vs M" when Qwen S5 > Nanbeige S5 and exact p < 0.05
#   the single-turn count and the agenttest pass count follow in the same sentence; a disagreement between them (S5 win but single-turn or agenttest loss beyond the margins) is said as MIXED, not hidden.
#   B (which-file, rule 1): Q4_K_M vs Q6_K Nanbeige on S5 (paired) + the measured ceiling table: "which file fits a 32 GB card at what context". C: the 64k line from the agenttest rows (f16 NOFIT / q8kv result, peak MiB, tier).
# SAMPLING: every board/toolcall/gpqa/aime row is GREEDY, as every Qwen comparator row. The Nanbeige card samples (agentic temp 1.0 / top_p 0.95; reasoning temp 0.6 / top_p 0.95) and evaluates at unlimited budget
#   with preserve_thinking=true: the readout says so wherever a score is set beside the card's. The agenttest rows use the card's agentic set as explicit flags (temp 1.0, top_p 0.95, top_k 0 = off, min_p 0.05 as the Spark precedent,
#   verified against /props before the first task), as the Qwen rows used theirs. preserve_thinking: left undefined, which keeps earlier reasoning, the same regime as the Qwen toolcall rows and the card's setting for tool use.
# NOT MEASURED: q8_0-KV accuracy on these tasks beyond the labelled rows; the card's own harnesses (Terminus 2, mini-swe-agent, Toolathlon); more than one run per leg.
#
# CARD MINUTES (measured Qwen3.8-27B leg times scaled by the decode ratio; Nanbeige Q4_K_M ~63 tok/s, Q6_K ~49 tok/s ESTIMATES from the Qwen Q4_K_M 78.94 / Q6_K 63.08 tg128 by weight bytes; token volume assumed equal to Qwen's,
#   Nanbeige may think longer, high case +40-50% on gpqa/aime): Q4_K_M fit 20 | speed 15 | board 105 | tc_st 85 | tc_s5 63 | agent 43 | gpqa 325 | aime 135 = 791 min (13.2 h)
#   Q6_K fit 14 | speed 20 | board 140 | tc_st 105 | tc_s5 77 | agent 6 | gpqa 415 | aime 170 = 947 min (15.8 h);  total ~1,740 min = 29 h (high case ~38 h). Details + sources in the vault status note.
#
# test knobs (all empty in production): NB_ONLY_LEGS="q4_fit" (restrict the leg list), NB_FIT_F16_POINTS_Q4 / NB_FIT_Q8_POINTS_Q4 / ..._Q6 / NB_FIT_REFINE (a reduced fit ladder for a sub-10-min smoke), NB_MIN_OVERRIDE (minutes-to-start floor, smoke only), NB_MANIFEST, NB_RUNLOG, NL_HARD_STOP_EPOCH (night-lib).
set -uo pipefail
NL_TAG=nanbeige43
MANIFEST=/home/witcheer/nanbeige43-status.json
export MANIFEST
. /home/witcheer/night-lib.sh
[ "${DRYRUN:-0}" = "1" ] && MANIFEST=/tmp/nanbeige43-dryrun.json   # a DRYRUN must never wipe the real manifest (the drain DRYRUNs before every block)
[ -n "${NB_MANIFEST:-}" ] && MANIFEST=$NB_MANIFEST
RUNLOG=${NB_RUNLOG:-/home/witcheer/nanbeige43-run.log}
RIG=/home/witcheer/benchmark-rig; PY=$RIG/.venv/bin/python
T=/home/witcheer/toolcall-v1; TPY=$T/.venv/bin/python; TC=$T/code-nb; TOK=$T/nanbeige-tokenizer
PROMPTS=$T/artifacts/nanbeige-prompts/prompts_toolcall-v1.jsonl.gz; TMAN=$T/frozen/toolcall-v1/manifest.json
NBD=/home/witcheer/nanbeige43
AT=/home/witcheer/agenttest-v2/code; AT_RES=/home/witcheer/agenttest-v2/results/nb43; AT_WORK=/home/witcheer/agenttest-v2/work/nb43
MAIN=/home/witcheer/llama.cpp-nl/build/bin/llama-server; BENCH=/home/witcheer/llama.cpp-nl/build/bin/llama-bench; MAIN_PIN=5d16e81
BOLT=/home/witcheer/llama.cpp-bolt/build/bin/llama-server; BOLT_PIN=cd0a5b6a1
HERMES=/home/witcheer/.local/bin/hermes; HERMES_PIN=${L1_HERMES_PIN:-9bda188}
GGUFPY=/home/witcheer/llama.cpp/gguf-py
MD=/home/witcheer/models/nanbeige43-33b
OVR=(--override-kv tokenizer.ggml.add_space_prefix=bool:false --fit off)
declare -A FILE=([q4]=Nanbeige_Nanbeige4.3-33B-Q4_K_M.gguf [q6]=Nanbeige_Nanbeige4.3-33B-Q6_K.gguf)
declare -A BYTES=([q4]=20788825216 [q6]=29016857728)
declare -A SLUG=([q4]=nanbeige-nanbeige4-3-33b-q4-k-m [q6]=nanbeige-nanbeige4-3-33b-q6-k)
declare -A ROWBASE=([q4]=q4km [q6]=q6k)
declare -A MIN_MIN=([fit]=25 [speed]=35 [board5]=25 [tc_st]=20 [tc_s5]=20 [agent]=25 [gpqa]=30 [aime]=30)   # minutes a leg needs before it may start (resumable legs: a floor; fit/speed: their whole length)
declare -A EST=([q4_fit]=20 [q4_speed]=15 [q4_board5]=105 [q4_tc_st]=85 [q4_tc_s5]=63 [q4_agent]=43 [q4_tc_st_stocktok]=85 [q4_gpqa]=325 [q4_aime]=135 [q6_fit]=14 [q6_speed]=20 [q6_board5]=140 [q6_tc_st]=105 [q6_tc_s5]=77 [q6_agent]=6 [q6_gpqa]=415 [q6_aime]=170)
CTX_AGENT=65536; PORT_TC=8097; PORT_AT=8177; PORT_FIT=8199
mkdir -p "$NBD/tomb" "$AT_RES" "$AT_WORK"

LEGS=(); for f in q4 q6; do for l in fit speed board5 tc_st tc_s5 agent gpqa aime; do
  [ "$f" = q6 ] && { [ "$l" = gpqa ] || [ "$l" = aime ]; } && [ -f /home/witcheer/nanbeige43-q6-long.disable ] && continue
  LEGS+=("${f}_${l}"); done; done
if [ -n "${NB_ONLY_LEGS:-}" ]; then L2=(); for x in "${LEGS[@]}"; do case " $NB_ONLY_LEGS " in *" $x "*) L2+=("$x");; esac; done; LEGS=("${L2[@]}"); fi
[ -f /home/witcheer/nanbeige43-extra.enable ] && LEGS+=(q4_tc_st_stocktok)   # EXTRA: Q4_K_M st-on with the STOCK GGUF tokenizer (no override), text prompts through the same driver: the cost of the llama.cpp space-prefix quirk, paired against the fixed leg

fl() { echo "${1%%_*}"; }; ll() { echo "${1#*_}"; }; base_of() { local l; l=$(ll "$1"); echo "${l%_stocktok}"; }
row_of() { echo "nanbeige43-33b-${ROWBASE[$1]}-$2"; }       # <f> f16|q8kv
agent_done_stamp() { echo "$AT_RES/${ROWBASE[$1]}.leg.done"; }
art_of() { local f l; f=$(fl "$1"); l=$(ll "$1"); local s=${SLUG[$f]}
  case "$l" in
    fit) echo "$RIG/results/$s/fit/fit.json";;
    speed) echo "$RIG/results/$s/speed.json";;
    board5) echo "$RIG/results/$s/quality.json";;
    tc_st) echo "$RIG/results/toolcall-v1/$s-st-on/summary.json";;
    tc_s5) echo "$RIG/results/toolcall-v1/$s-s5-on/summary.json";;
    agent) agent_done_stamp "$f";;
    gpqa) echo "$RIG/results/$s-thinkon/gpqa.json";;
    aime) echo "$RIG/results/$s-matharena-32k/ma_aime_2026_detail.json";;
    tc_st_stocktok) echo "$RIG/results/toolcall-v1/$s-st-on-stocktok/summary.json";;
  esac; }
tomb_of() { echo "$NBD/tomb/$1"; }
leg_done() { [ -f "$(art_of "$1")" ] || [ -f "$(tomb_of "$1")" ]; }

# ---------------------------------------------------------------- PLAN (no GPU: every probe runs with the card hidden)
nl_say "PLAN phase (DRYRUN=${DRYRUN:-0}, legs ${#LEGS[@]})"
nl_check_path "$PY" "$TPY" "$MAIN" "$BENCH" "$BOLT" "$HERMES" "$GGUFPY/gguf/__init__.py" "$NBD/nb_fit.py" "$NBD/nb_speed.py" "$NBD/nb_cell.py" "$NBD/nb_plan.py" "$NBD/nb_readout.py" "$NBD/rows-nanbeige.json" \
  "$RIG/lib/quality.py" "$RIG/lib/evals/gpqa.py" "$RIG/lib/evals/matharena.py" "$TC/driver.py" "$TC/qwen38_xml.py" "$TOK/tokenizer.json" "$TOK/tokenizer.model" "$TMAN" "$PROMPTS" "$AT/run_agent.py" "$AT/row_info.py" "$AT/aggregate.py" "$AT/parse_server_log.py"
for f in q4 q6; do
  p=$MD/${FILE[$f]}
  if [ ! -f "$p" ]; then nl_fail "gguf MISSING: $p"; continue; fi
  got=$(stat -c %s "$p"); [ "$got" = "${BYTES[$f]}" ] && nl_ok "${FILE[$f]} $got B" || nl_fail "${FILE[$f]} is $got B, want ${BYTES[$f]}"
  st="$got $(stat -c %Y "$p")"; want=$(grep -F "${FILE[$f]}" "$NBD/gguf.stat" 2>/dev/null | cut -d' ' -f2-3)
  [ "$st" = "$want" ] && nl_ok "${FILE[$f]} unchanged since the sha256 check (stat stamp)" || nl_fail "${FILE[$f]} size/mtime differs from $NBD/gguf.stat (re-verify sha256, then re-stamp)"
done
ls "$MD"/.cache/huggingface/download/*.incomplete >/dev/null 2>&1 && nl_fail "hf .incomplete present in $MD"
case "$(CUDA_VISIBLE_DEVICES= $MAIN --version 2>&1)" in *"$MAIN_PIN"*) nl_ok "board build llama-server $MAIN_PIN (b10371, config.yaml server_bin)";; *) nl_fail "board llama-server build changed (want $MAIN_PIN)";; esac
case "$(CUDA_VISIBLE_DEVICES= $BOLT --version 2>&1)" in *"$BOLT_PIN"*) nl_ok "agenttest build llama-server $BOLT_PIN (b11668)";; *) nl_fail "bolt llama-server build changed (want $BOLT_PIN)";; esac
for B in "$MAIN" "$BOLT"; do HELP=$(CUDA_VISIBLE_DEVICES= $B --help 2>&1 || true)
  for want in "--override-kv" "--fit " "--cache-type-k" "--flash-attn" "--jinja" "--min-p" "--top-k" "--log-timestamps"; do case "$HELP" in *"$want"*) :;; *) nl_fail "$(basename $(dirname $(dirname $(dirname $B)))) llama-server lacks $want";; esac; done; done
BH=$(CUDA_VISIBLE_DEVICES= $BENCH --help 2>&1 || true); for want in "--cache-type-k" "--n-depth" "--flash-attn"; do case "$BH" in *"$want"*) :;; *) nl_fail "llama-bench lacks $want";; esac; done
nl_ok "servers and llama-bench list the flags this item passes (override-kv, fit, cache-type-k/v, flash-attn, n-depth)"
# gguf header: arch, shape, tokenizer keys, embedded template == HF template
for f in q4 q6; do H=$(nice -n 10 env CUDA_VISIBLE_DEVICES= $PY - "$MD/${FILE[$f]}" "$GGUFPY" <<'PYEOF' 2>&1
import sys, hashlib
sys.path.insert(0, sys.argv[2])
from gguf import GGUFReader
f = GGUFReader(sys.argv[1]).fields
g = lambda k: f[k].contents()
assert g("general.architecture") == "llama" and g("llama.block_count") == 52 and g("llama.embedding_length") == 7168 and g("llama.attention.head_count_kv") == 8 and g("llama.vocab_size") == 166912
assert g("tokenizer.ggml.add_bos_token") is True and g("tokenizer.ggml.add_space_prefix") is True and g("tokenizer.ggml.bos_token_id") == 166100
t = g("tokenizer.chat_template")
print("hdr-ok", hashlib.sha256(t.encode()).hexdigest())
PYEOF
); case "$H" in "hdr-ok f2b1e2a1be25d277e8c903a2aa4d69ca5fd8a0fcb37d4792088e72426df5b91c") nl_ok "${FILE[$f]} header: arch llama 52 blocks, add_bos + add_space_prefix true (the two traps), embedded template sha f2b1e2a1 = the HF template";; *) nl_fail "${FILE[$f]} header check: $(echo "$H" | tail -2 | tr '\n' ' ')";; esac; done
# toolcall-v1 pieces
( cd $T/frozen/toolcall-v1 && sha256sum -c --quiet FROZEN.sha256 ) && nl_ok "toolcall frozen list sha256 intact" || nl_fail "toolcall frozen list changed"
( cd $TOK && sha256sum -c --quiet SHA256SUMS.txt ) && nl_ok "nanbeige tokenizer files match SHA256SUMS (HF df919e1e)" || nl_fail "nanbeige tokenizer files changed"
( cd $TC && $TPY -c "import qwen38_xml as X; assert X.PARSER_VERSION == 'toolcall-v1-parser-2', X.PARSER_VERSION" 2>/dev/null ) && nl_ok "parser toolcall-v1-parser-2" || nl_fail "parser version changed (re-run roundtrip_test.py)"
grep -q -- "--prompt-tokens" $TC/driver.py && nl_ok "driver has --prompt-tokens" || nl_fail "driver lacks --prompt-tokens"
[ "$(jq -r '.roundtrip_failures // empty' $T/artifacts/nanbeige-roundtrip.json 2>/dev/null || python3 -c "import json;print(json.load(open('$T/artifacts/nanbeige-roundtrip.json'))['roundtrip_failures'])")" = 0 ] && nl_ok "nanbeige roundtrip artifact: 799/799, 0 failures (parser unchanged)" || nl_fail "nanbeige roundtrip artifact missing or has failures"
( cd $TC && $TPY -c "from bfcl_eval.utils import load_ground_truth_entry; import bfcl_items" 2>/dev/null ) && nl_ok "bfcl-eval imports in the toolcall venv" || nl_fail "bfcl-eval import failed"
# agenttest harness (as 73): frozen set, proven checkers, Hermes pin, rows file
( cd "$AT" && sha256sum -c --quiet FROZEN.sha256 ) && nl_ok "agenttest frozen task set sha256 intact" || nl_fail "agenttest frozen task set changed"
FZ=$(cd "$AT" && AT2_STRICT=1 python3 freeze.py --check 2>&1 || true); case "$FZ" in *"matches the registry"*) nl_ok "agenttest tasks.json = registry + seed hashes";; *) nl_fail "agenttest tasks.json mismatch: $FZ";; esac
PROOF=$(cd "$AT" && python3 prove_checkers.py --work /home/witcheer/agenttest-v2/at2-proof-nb --out /tmp/at2-proof-nb.json 2>&1 | tail -1 || true); case "$PROOF" in "ALL_OK 30/30"*) nl_ok "agenttest checker proof on this box: $PROOF";; *) nl_fail "checker proof failed: $PROOF";; esac
case "$($HERMES --version 2>&1)" in *"$HERMES_PIN"*) nl_ok "Hermes Agent pin $HERMES_PIN";; *) nl_fail "Hermes version changed (want $HERMES_PIN)";; esac
mapfile -t ROWS < <(python3 "$AT/row_info.py" --rows "$NBD/rows-nanbeige.json"); [ "${#ROWS[@]}" = 4 ] && nl_ok "rows-nanbeige.json: 4 rows" || nl_fail "rows-nanbeige.json gives ${#ROWS[@]} rows, want 4"
nl_check_disk 30
free_g=$(free -g | awk '/^Mem:/{print $2}'); [ "$free_g" -ge 55 ] && nl_ok "RAM ${free_g}G" || nl_fail "RAM ${free_g}G < 55"
$PY -c "from datasets import load_dataset; assert len(load_dataset('Idavidrein/gpqa','gpqa_diamond',split='train'))==198" >/dev/null 2>&1 && nl_ok "gpqa_diamond loads (198)" || nl_fail "gpqa dataset probe FAILED (gated: the HF account must hold the grant)"
$PY -c "from datasets import load_dataset; assert len(load_dataset('MathArena/aime_2026',split='train'))==30" >/dev/null 2>&1 && nl_ok "MathArena/aime_2026 loads (30)" || nl_fail "aime_2026 dataset probe FAILED"
(cd $RIG && $PY -c "from lib.qwen_math_grader import grade; assert grade(r'\\frac{1}{2}','0.5')") >/dev/null 2>&1 && nl_ok "math grader imports" || nl_fail "math grader broken"
$PY -c "import sys; sys.path.insert(0,'$RIG'); from latex2sympy2_extended import latex2sympy; from lib.evals.matharena import grade_with_timeout as g; assert g('277','277', veto=True)[0]" >/dev/null 2>&1 && nl_ok "exact-number veto" || nl_fail "exact-number veto check FAILED"
NL_SELFTEST=1 CUDA_VISIBLE_DEVICES= PYTHONPATH="$RIG" $PY - <<'PYEOF' >/dev/null 2>&1 && nl_ok "harness self-test: eval classes construct; nb_cell/nb_speed/nb_fit compile" || nl_fail "harness self-test FAILED"
import tempfile, py_compile
from pathlib import Path
from lib.evals.base import LLMClient
from lib.evals.gpqa import GPQAEval
from lib.evals.matharena import MathArenaEval
rd = Path(tempfile.mkdtemp()); c = LLMClient("http://127.0.0.1:1/v1", "x", think=True, timeout=5)
GPQAEval(client=c, results_dir=rd, max_tokens=16384); MathArenaEval(client=c, task="ma_aime_2026", results_dir=rd, max_tokens=32768, exact_veto=True)
for f in ("nb_fit", "nb_speed", "nb_cell", "nb_plan", "nb_readout"): py_compile.compile(f"/home/witcheer/nanbeige43/{f}.py", doraise=True)
PYEOF
# plan: one RUN/SKIP line per leg with the literal artifact path
PH=""; RUNS=0; EST_MIN=0
for lg in "${LEGS[@]}"; do PH="$PH $lg"; a=$(art_of "$lg")
  if leg_done "$lg"; then nl_say "  $lg: SKIP, $( [ -f "$a" ] && echo "artifact present ($a)" || echo "tombstone $(tomb_of "$lg")")"
  else nl_say "  $lg: RUN (est ${EST[$lg]:-60} min, needs ${MIN_MIN[$(base_of "$lg")]} min to start; artifact $a)"; RUNS=$((RUNS+1)); EST_MIN=$((EST_MIN+${EST[$lg]:-60})); fi; done
DL=$(nl_deadline_epoch); nl_say "  planned RUN: $RUNS legs (est $EST_MIN card min = $((EST_MIN/60)) h $((EST_MIN%60)) min); hard stop $(date -d "@$DL" '+%a %H:%M') local, $(( (DL-$(date +%s))/60 )) min away"
[ -z "${EXPECT_RUN:-}" ] || [ "$RUNS" = "$EXPECT_RUN" ] || { nl_say "ABORT: planned RUN $RUNS != EXPECT_RUN $EXPECT_RUN"; exit 2; }
[ "$RUNS" -gt 0 ] || { nl_say "nothing to run, item complete"; exit 0; }
if [ "${DRYRUN:-0}" = "1" ] || [ ! -f "$MANIFEST" ]; then nl_manifest_init $PH; fi
nl_gate

# ---------------------------------------------------------------- GPU phase
SP=""; GP=""; GP2=""; DRAINED=0
vram() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' '; }
wait_vram_free() { local i v; for i in $(seq 1 60); do v=$(vram); [ "${v:-99999}" -lt 1100 ] && return 0; sleep 2; done; return 1; }
stop_server() { # own pid only, never pkill: a sibling's llama-server must survive
  [ -n "$SP" ] && kill "$SP" 2>/dev/null; sleep 3; [ -n "$SP" ] && kill -0 "$SP" 2>/dev/null && { kill -9 "$SP" 2>/dev/null; sleep 2; }
  SP=""; wait_vram_free || nl_say "  WARN: VRAM still $(vram) MiB after server stop"; }
kill_my_servers() { # a leg's python child may leave its llama-server behind: kill only compute pids whose cmdline names a Nanbeige gguf
  local pid; for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
    tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | grep -q "/models/nanbeige43-33b/" && { nl_say "  killing leftover server pid $pid (names a Nanbeige gguf)"; kill "$pid" 2>/dev/null; sleep 4; kill -9 "$pid" 2>/dev/null; }; done
  wait_vram_free || nl_say "  WARN: VRAM still $(vram) MiB after leg cleanup"; }
stop_sampler() { [ -n "$GP" ] && kill "$GP" 2>/dev/null; [ -n "$GP2" ] && kill "$GP2" 2>/dev/null; GP=""; GP2=""; }
restore() { stop_sampler; stop_server
  if [ "$DRAINED" = 1 ]; then nl_say "EXIT trap: restoring Donald"; systemctl is-active --quiet llama-server.service || sudo -n systemctl start llama-server.service || true
    for i in $(seq 1 30); do curl -s -m 3 http://127.0.0.1:8090/health | grep -qi ok && { nl_say "Donald proxy health OK"; break; }; sleep 2; done; fi; }
trap restore EXIT
nl_say "draining Donald"; sudo -n systemctl stop llama-server.service; DRAINED=1; sleep 3
for i in $(seq 1 120); do v=$(vram); [ "${v:-99999}" -lt 1100 ] && { nl_say "drain clear (${v}MiB)"; break; }; sleep 2; [ "$i" -eq 120 ] && { nl_say "FATAL drain timeout (a sibling holds the card?)"; exit 1; }; done
DEADLINE=$(nl_deadline_epoch)
left_s() { echo $(( DEADLINE - $(date +%s) )); }
cd "$RIG"; export PYTHONPATH="$RIG"

leg_fail() { # <leg> <reason>: a persistent failure (not the window, not an OOM verdict) twice in a row becomes a tombstone so the queue does not retry it every block
  local n; n=$(( $(cat "$NBD/tomb/$1.fails" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$NBD/tomb/$1.fails"
  nl_say "  $1: FAILED ($2), failure $n of 2"; nl_mark "$1" failed "$2"
  if [ "$n" -ge 2 ]; then printf '{"leg":"%s","tombstone":true,"reason":"%s twice in a row"}\n' "$1" "$2" > "$(tomb_of "$1")"; nl_say "  $1: tombstoned after two failures (counts as done, reason on file)"; fi; }
leg_nofit() { printf '{"leg":"%s","nofit":true,"reason":"%s"}\n' "$1" "$2" > "$(tomb_of "$1")"; nl_say "  $1: NOFIT, $2"; nl_mark "$1" skipped "NOFIT: $2"; }
kvflags() { KVF=(); [ "$1" = q8_0 ] && KVF=(-fa on --cache-type-k q8_0 --cache-type-v q8_0); return 0; }
start_tc_server() { # <f> <ctx> <kv> <log> <bin> <noovr 0|1>
  local f=$1 ctx=$2 kv=$3 log=$4 bin=$5 noovr=${6:-0}; kvflags "$kv"; local ov=("${OVR[@]}"); [ "$noovr" = 1 ] && ov=(--fit off)
  "$bin" -m "$MD/${FILE[$f]}" --port $PORT_TC --n-gpu-layers 99 --host 127.0.0.1 --jinja --ctx-size "$ctx" "${ov[@]}" "${KVF[@]}" >> "$log" 2>&1 &
  SP=$!; local i; for i in $(seq 1 180); do curl -s -m 3 http://127.0.0.1:$PORT_TC/health | grep -qi ok && return 0; kill -0 $SP 2>/dev/null || break; sleep 2; done; return 1; }
tok_guard() { # the server must tokenise the template boundaries like the HF tokenizer
  python3 - $PORT_TC <<'PYEOF'
import json, sys, urllib.request
G = [("<|im_start|>user\nhi<|im_end|>\n", [166100, 2714, 13, 6687, 166101, 13]), ("<|im_start|>assistant\n<think>\nabc\n</think>\n\nhello", [166100, 66354, 13, 166103, 13, 21893, 13, 166104, 13, 13, 20255])]
for t, w in G:
    r = urllib.request.Request(f"http://127.0.0.1:{sys.argv[1]}/tokenize", data=json.dumps({"content": t, "add_special": False, "parse_special": True}).encode(), headers={"Content-Type": "application/json"})
    got = json.loads(urllib.request.urlopen(r, timeout=30).read())["tokens"]
    if got != w: print("GUARD-FAIL", got, w); sys.exit(1)
print("guard ok")
PYEOF
}

# ---- legs ----
run_fit() { local f=$1 s=${SLUG[$1]}; local fd="$RIG/results/$s/fit"; mkdir -p "$fd"; local kv rc=0
  for kv in f16 q8_0; do
    [ "$(python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get('complete'))" "$fd/fit-$kv.json" 2>/dev/null)" = True ] && { nl_say "  fit $f $kv: complete on file"; continue; }
    local pts=() pv="NB_FIT_$([ "$kv" = f16 ] && echo F16 || echo Q8)_POINTS_${f^^}"; [ -n "${!pv:-}" ] && pts=(--points "${!pv}")   # test knob, e.g. NB_FIT_F16_POINTS_Q4="49152 57344 65536"
    timeout -k 30 $(( $(left_s) - 60 )) python3 -u "$NBD/nb_fit.py" --server-bin "$MAIN" --model "$MD/${FILE[$f]}" --kv "$kv" --out "$fd/fit-$kv.json" --port $PORT_FIT --deadline-epoch "$DEADLINE" --label "$s" --log-dir "$fd" --refine "${NB_FIT_REFINE:-2}" "${pts[@]}" >> "$RUNLOG" 2>&1; rc=$?
    kill_my_servers
    [ "$rc" = 3 ] || [ "$rc" = 124 ] && { nl_say "  fit $f $kv: ladder paused at the hard stop (resumable)"; return 3; }
    [ "$rc" = 0 ] || return 1
  done
  python3 "$NBD/nb_plan.py" --fitdir "$fd" --merge >> "$RUNLOG" 2>&1; [ -f "$fd/fit.json" ] && [ "$(python3 -c "import json;print(json.load(open('$fd/fit.json'))['complete'])")" = True ] && return 0; return 3; }

run_speed() { local f=$1 s=${SLUG[$1]}; local nv; nv=$(timeout -k 30 $(( $(left_s) - 60 )) env PYTHONPATH="$RIG" "$PY" -u "$NBD/nb_speed.py" --model "$MD/${FILE[$f]}" --bench-bin "$BENCH" --results-dir "$RIG/results" --build-label "b10371 5d16e81dd (llama.cpp-nl)" >> "$RUNLOG" 2>&1; echo $?); kill_my_servers; return "$nv"; }

run_cell() { # <leg> <f> <cell> <need ctx> <min ctx> [extra nb_cell args]
  local leg=$1 f=$2 cell=$3 need=$4 min=$5; shift 5; local s=${SLUG[$f]}
  PREST=$(python3 "$NBD/nb_plan.py" --fitdir "$RIG/results/$s/fit" --need "$need" --min "$min"); nl_say "  plan $f $cell: $PREST"
  case "$PREST" in OK*) ;; "NOFIT fit leg not done"*) nl_say "  $leg: fit leg not complete, resumable"; return 3;; *) leg_nofit "$leg" "$PREST"; return 9;; esac
  local _ kv ctx lim; read -r _ kv ctx lim <<< "$PREST"
  timeout -k 30 $(( $(left_s) - 90 )) "$PY" -u "$NBD/nb_cell.py" --model "$MD/${FILE[$f]}" --slug "$s" --cell "$cell" --ctx "$ctx" --kv "$kv" --build-label "b10371 5d16e81dd" --file-label "bartowski/Nanbeige_Nanbeige4.3-33B-GGUF ${FILE[$f]} (${BYTES[$f]} B)" "$@" >> "$RUNLOG" 2>&1
  local rc=$?; kill_my_servers; cp -f ~/bench-server.log "$RIG/results/$s/server-$cell.log" 2>/dev/null || true; return $rc; }

run_tc() { # <leg> <f> <st|s5> [stocktok]
  local leg=$1 f=$2 mode=$3 stock=${4:-0} s=${SLUG[$2]}; local need min sets suffix
  if [ "$mode" = st ]; then need=24576; min=12288; sets=S1,S2,S3,S4; suffix=st-on; else need=40960; min=24576; sets=S5; suffix=s5-on; fi
  [ "$stock" = 1 ] && suffix="$suffix-stocktok"
  PREST=$(python3 "$NBD/nb_plan.py" --fitdir "$RIG/results/$s/fit" --need "$need" --min "$min"); nl_say "  plan $f tc_$mode: $PREST"
  case "$PREST" in OK*) ;; "NOFIT fit leg not done"*) nl_say "  $leg: fit leg not complete, resumable"; return 3;; *) leg_nofit "$leg" "$PREST"; return 9;; esac
  local _ kv ctx lim; read -r _ kv ctx lim <<< "$PREST"
  local d="$RIG/results/toolcall-v1/$s-$suffix"; mkdir -p "$d"; echo "$PREST" > "$d/plan.txt"
  local left; left=$(( $(left_s) - 120 ))
  start_tc_server "$f" "$ctx" "$kv" "$d/server.log" "$MAIN" "$stock"; local up=$?
  if [ "$up" != 0 ]; then stop_server; leg_fail "$leg" "server did not come up at ctx $ctx kv $kv"; return 1; fi
  if [ "$stock" = 0 ]; then tok_guard >> "$RUNLOG" 2>&1 || { stop_server; leg_fail "$leg" "tokenizer guard failed (override-kv not applied)"; return 1; }; fi
  local pt=(--prompt-tokens); [ "$stock" = 1 ] && pt=()
  ( cd "$TC" && timeout -k 30 $(( left + 60 )) "$TPY" -u driver.py --server http://127.0.0.1:$PORT_TC --label "$s-$suffix" --sets "$sets" --manifest "$TMAN" --tokenizer "$TOK" --prompts "$PROMPTS" \
      --out "$d" --think on --max-tokens 16384 --ctx "$ctx" --budget-s "$left" --tokenize-all "${pt[@]}" --template-label "Nanbeige4.3 HF template (== GGUF embedded, sha f2b1e2a1), kv $kv, tokenizer fix $([ "$stock" = 1 ] && echo OFF-stock || echo add_space_prefix=false)" \
      --server-bin "llama.cpp b10371 5d16e81dd" >> "$RUNLOG" 2>&1 ); local rc=$?
  stop_server; return $rc; }

# ---- agenttest leg: functions copied from 73-agenttest-v2-c1.sh (the proven invocation), rows file and results dir changed, KV flags + tokenizer override added ----
CTX=$CTX_AGENT; RES=$AT_RES; WORK=$AT_WORK; CODE=$AT; PORT=$PORT_AT; SRV=$BOLT; MODELS=/home/witcheer/models; LOADWAIT=${AT2_LOADWAIT:-900}; KVX=(); KVLABEL="f16 KV"
done_row() { [ -f "$RES/$1/summary.json" ] || [ -f "$RES/$1/load-failed.json" ]; }
row_fail() { local d="$RES/$1" n; n=$(( $(cat "$d/fails" 2>/dev/null || echo 0) + 1 )); echo "$n" > "$d/fails"
  nl_say "  $1: FAILED ($2), failure $n of 2"; nl_mark "$1" failed "$2"
  if [ "$n" -ge 2 ]; then printf '{"label":"%s","load_failed":true,"reason":"%s twice in a row"}\n' "$1" "$2" > "$RES/$1/load-failed.json"; nl_say "  $1: recorded as NOFIT after two failures"; return 0; fi
  return 3; }
start_server() { # <path> <temp> <top_p> <top_k> <min_p> <moe> <log> -> 0 up, 1 failed, 4 OOM signature
  local path=$1 temp=$2 tp=$3 tk=$4 mp=$5 moe=$6 log=$7
  "$SRV" -m "$MODELS/$path" -ngl 99 -fa on -c $CTX --jinja --host 127.0.0.1 --port "$PORT" --log-timestamps \
     --temp "$temp" --top-p "$tp" --top-k "$tk" --min-p "$mp" "${OVR[@]}" "${KVX[@]}" >> "$log" 2>&1 &
  SP=$!
  local i; for i in $(seq 1 $((LOADWAIT/2))); do curl -s -m 3 http://127.0.0.1:$PORT/health | grep -qi ok && break; kill -0 $SP 2>/dev/null || break; sleep 2; done
  if curl -s -m 3 http://127.0.0.1:$PORT/health | grep -qi ok; then return 0; fi
  local oom=1; tail -300 "$log" | grep -qiE "out of memory|failed to allocate|cudaMalloc failed|ggml_backend_cuda_buffer_type_alloc_buffer|unable to allocate|failed to create context" && oom=4
  stop_server; return $oom; }
effort_probe() {
  local c
  c=$(curl -s -o /tmp/nb-probe.json -w '%{http_code}' -m 180 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"x","messages":[{"role":"user","content":"Say ok."}],"reasoning_effort":"medium","max_tokens":24}')
  if [ "$c" = 200 ]; then echo unset; return; fi
  c=$(curl -s -o /tmp/nb-probe.json -w '%{http_code}' -m 180 http://127.0.0.1:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"x","messages":[{"role":"user","content":"Say ok."}],"reasoning_effort":"high","max_tokens":24}')
  [ "$c" = 200 ] && echo high || echo fail; }
run_row() { # <line> -> 0 done/NOFIT, 3 resumable
  IFS='|' read -r slug path bytes arch est temp top_p top_k min_p moe prof label <<< "$1"
  local d="$RES/$slug" t0 left rc attempt effort
  left=$(left_s)
  [ "$left" -gt 900 ] || { nl_say "window exhausted before $slug ($left s left), resumable"; nl_mark "$slug" failed "window exhausted (resumable)"; return 3; }
  mkdir -p "$d" "$WORK/$slug"; t0=$(date +%s); nl_mark "$slug" running; nl_say "=== $slug ($label, est $est min, budget $((left/60)) min, $KVLABEL) ==="
  start_server "$path" "$temp" "$top_p" "$top_k" "$min_p" "$moe" "$d/server.log"; rc=$?
  if [ "$rc" = 4 ]; then
    printf '{"label":"%s","load_failed":true,"reason":"out-of-memory signature in the server log at -c %s with %s (card %s MiB)","vram_mib_now":%s}\n' "$slug" "$CTX" "$KVLABEL" "$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')" "$(vram)" > "$d/load-failed.json"
    nl_say "  $slug: DOES NOT FIT at ctx $CTX with $KVLABEL (out of memory at load), recorded as NOFIT"; nl_mark "$slug" skipped "NOFIT"; return 0
  elif [ "$rc" != 0 ]; then row_fail "$slug" "server did not come up (not an OOM signature)"; return $?; fi
  local vl; vl=$(vram)
  nl_say "  server up, VRAM $vl MiB"
  curl -s -m 10 http://127.0.0.1:$PORT/props > "$d/props.json" 2>/dev/null
  effort=$(effort_probe); nl_say "  reasoning_effort probe: $effort"
  if [ "$effort" = fail ]; then stop_server; row_fail "$slug" "both reasoning_effort medium and high rejected"; return $?; fi
  python3 - "$d/props.json" "$d/row-meta.json" "$slug" "$label" "$path" "$bytes" "$temp" "$top_p" "$top_k" "$min_p" "-" "$effort" "$vl" "$HERMES_PIN" "$($SRV --version 2>&1 | grep -i version | head -1)" "$($HERMES --version 2>&1 | head -1)" <<'PY'
import json, sys
props, out, slug, label, path, nbytes, temp, tp, tk, mp, moe, effort, vl, pin, build, hv = sys.argv[1:17]
p = json.load(open(props)) if props else {}
g = (p.get("default_generation_settings") or {}).get("params") or {}
applied = {k: g.get(k) for k in ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repeat_penalty", "n_predict")}
want = {"temperature": float(temp), "top_p": float(tp), "top_k": int(tk), "min_p": float(mp)}
ok = all(applied.get(k) is not None and abs(float(applied[k]) - v) < 1e-5 for k, v in want.items() if not (k == "top_k" and int(tk) <= 0))
json.dump({"row": slug, "label": label, "path": path, "bytes": int(nbytes), "ctx": p.get("default_generation_settings", {}).get("n_ctx"), "total_slots": p.get("total_slots"),
           "sampling_flags": want, "sampling_applied": applied, "sampling_match": ok, "n_cpu_moe_used": None, "effort": effort, "vram_loaded_mib": int(vl),
           "llama_build": build, "hermes_version": hv, "hermes_pin": pin, "template_mentions_reasoning_effort": "reasoning_effort" in (p.get("chat_template") or "")}, open(out, "w"), indent=1)
print("row-meta: sampling flags vs /props applied:", "MATCH" if ok else "MISMATCH", applied)
sys.exit(0 if ok else 5)
PY
  rc=$?
  if [ "$rc" != 0 ]; then stop_server; row_fail "$slug" "sampling flags do not match what /props reports"; return $?; fi
  nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits -l 2 >> "$d/gpu-samples.csv" 2>/dev/null & GP=$!
  nvidia-smi --query-compute-apps=timestamp,pid,process_name,used_memory --format=csv,noheader,nounits -l 2 >> "$d/gpu-apps.csv" 2>/dev/null & GP2=$!; echo "$SP" > "$d/server.pid"
  for attempt in 1 2 3; do
    python3 -u "$CODE/run_agent.py" --row "$slug" --profile "$prof" --model-id "$slug" --port "$PORT" --work-root "$WORK/$slug" --out "$d" --effort "$effort" --deadline-epoch "$DEADLINE" \
       --hermes "$HERMES" --ctx $CTX --meta "$d/row-meta.json" --gpu-samples "$d/gpu-samples.csv" >> "$RUNLOG" 2>&1
    rc=$?
    [ "$rc" = 4 ] || break
    nl_say "  $slug: server unhealthy mid-row (attempt $attempt), restarting it"; stop_server
    start_server "$path" "$temp" "$top_p" "$top_k" "$min_p" "$moe" "$d/server.log"; [ $? = 0 ] || { nl_say "  restart failed"; break; }
  done
  stop_sampler
  SKIP_FIRST=1 python3 "$CODE/parse_server_log.py" "$d/server.log" > "$d/server-timings.json" 2>/dev/null || true
  stop_server
  if [ "$rc" = 3 ]; then nl_say "  $slug: partial, resumable ($(ls "$d"/t*.result.json 2>/dev/null | wc -l) tasks done)"; nl_mark "$slug" failed "partial (resumable)"; return 3; fi
  if nl_judge_artifact "$slug" "$rc" "$d/summary.json" "$t0" "$(( ($(date +%s)-t0)/60 )) min"; then nl_say "  $(grep -a 'ROW-SUMMARY' "$RUNLOG" | tail -1)"; return 0
  else nl_say "  $slug: FAILED rc=$rc"; return 3; fi; }
fit_ceiling() { python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2]) or 0)" "$RIG/results/${SLUG[$1]}/fit/fit.json" "$2" 2>/dev/null || echo -1; }
mark_nofit_from_fit() { # <row> <kv label> <ceiling> -- the measured fit leg already says this KV type cannot reach 65536
  mkdir -p "$RES/$1"; printf '{"label":"%s","load_failed":true,"reason":"not tried: the fit leg measured a %s ceiling of %s tokens, below the 65536 Hermes needs (fit.json, board build b10371)"}\n' "$1" "$2" "$3" > "$RES/$1/load-failed.json"
  nl_say "  $1: NOFIT from the fit leg ($2 ceiling $3 < $CTX), not tried"; nl_mark "$1" skipped "NOFIT (fit leg: $2 ceiling $3)"; }
run_agent_leg() { local f=$1 r16 r8 line c16 c8
  r16=$(row_of "$f" f16); r8=$(row_of "$f" q8kv)
  c16=$(fit_ceiling "$f" ceiling_f16); c8=$(fit_ceiling "$f" ceiling_q8_0)
  if ! done_row "$r16" && [ "$c16" -ge 0 ] && [ "$c16" -lt "$CTX" ]; then mark_nofit_from_fit "$r16" "f16 KV" "$c16"; fi
  if ! done_row "$r8" && [ ! -f "$RES/$r16/summary.json" ] && [ "$c8" -ge 0 ] && [ "$c8" -lt "$CTX" ]; then mark_nofit_from_fit "$r8" "q8_0 KV" "$c8"; fi
  if ! done_row "$r16"; then
    line=$(python3 "$CODE/row_info.py" --rows "$NBD/rows-nanbeige.json" --only "$r16"); KVX=(); KVLABEL="f16 KV"
    run_row "$line" || return 3; done_row "$r16" || return 3
  fi
  if [ -f "$RES/$r16/summary.json" ]; then touch "$(agent_done_stamp "$f")"; return 0; fi
  # f16 is NOFIT at 64k: the q8kv row is the measurable one
  if ! done_row "$r8"; then
    line=$(python3 "$CODE/row_info.py" --rows "$NBD/rows-nanbeige.json" --only "$r8"); KVX=(-fa on --cache-type-k q8_0 --cache-type-v q8_0); KVLABEL="q8_0 K+V KV"
    run_row "$line" || return 3; done_row "$r8" || return 3
  fi
  touch "$(agent_done_stamp "$f")"; return 0; }

# ---------------------------------------------------------------- the leg loop
for lg in "${LEGS[@]}"; do
  f=$(fl "$lg"); l=$(ll "$lg"); base=${l%_stocktok}
  if leg_done "$lg"; then nl_mark "$lg" skipped "artifact present"; continue; fi
  left=$(left_s); need_min=${NB_MIN_OVERRIDE:-${MIN_MIN[$base]:-30}}   # NB_MIN_OVERRIDE: smoke knob only
  if [ "$left" -lt $(( need_min * 60 )) ]; then nl_say "window low before $lg ($((left/60)) min left, needs $need_min): block over for this item, resumable"; nl_mark "$lg" failed "window low (resumable)"; break; fi
  t0=$(date +%s); nl_mark "$lg" running; nl_say "=== $lg (est ${EST[$lg]:-60} min, budget $((left/60)) min) ==="
  case "$l" in
    fit) run_fit "$f"; rc=$?;;
    speed) run_speed "$f"; rc=$?;;
    board5) run_cell "$lg" "$f" board5 8192 8192; rc=$?;;
    tc_st) run_tc "$lg" "$f" st; rc=$?;;
    tc_s5) run_tc "$lg" "$f" s5; rc=$?;;
    tc_st_stocktok) run_tc "$lg" "$f" st 1; rc=$?;;
    agent) run_agent_leg "$f"; rc=$?;;
    gpqa) run_cell "$lg" "$f" gpqa_on 24576 20480; rc=$?;;
    aime) run_cell "$lg" "$f" ma_aime_2026 40960 34816; rc=$?;;
  esac
  [ "$rc" = 9 ] && continue   # NOFIT already recorded
  if [ "$l" = fit ] && [ "$rc" != 0 ]; then nl_say "  $lg: the ceilings decide every later leg of this file, stopping here (resumable)"; nl_mark "$lg" failed "partial or failed (resumable)"; break; fi
  art=$(art_of "$lg")
  if [ "$rc" = 3 ] || [ "$rc" = 124 ]; then nl_say "  $lg: partial, resumable at the hard stop"; nl_mark "$lg" failed "partial (resumable)"; continue; fi
  if nl_judge_artifact "$lg" "$rc" "$art" "$t0" "$(( ($(date +%s)-t0)/60 )) min"; then nl_say "  $lg: done in $(( ($(date +%s)-t0)/60 )) min"; rm -f "$NBD/tomb/$lg.fails"
  else leg_fail "$lg" "rc=$rc, artifact missing or stale"; fi
done

# ---- CPU tail: aggregate the agenttest rows, write the readout (cheap, idempotent, safe on a partial run) ----
python3 "$AT/aggregate.py" --res "$AT_RES" --out "$AT_RES" --rows "$NBD/rows-nanbeige.json" > "$AT_RES/aggregate.log" 2>&1 && nl_say "agenttest aggregate: $AT_RES/l1-readout.md" || nl_say "agenttest aggregate failed (see $AT_RES/aggregate.log)"
python3 "$NBD/nb_readout.py" --out "$NBD" > "$NBD/readout.log" 2>&1 && nl_say "readout: $NBD/readout.md (line 1: $(python3 -c "import json;print(json.load(open('$NBD/readout.json'))['line1'][:200])" 2>/dev/null))" || nl_say "readout failed (see $NBD/readout.log)"
restore; trap - EXIT
ok=0; for _ in $(seq 1 60); do curl -s -m 3 localhost:8090/health | grep -qi ok && { ok=1; break; }; sleep 5; done
[ "$ok" = 1 ] && nl_say "Donald healthy at end" || { nl_say "DONALD NOT HEALTHY, restoring"; nl_donald_restore; }
for lg in "${LEGS[@]}"; do leg_done "$lg" || { nl_say "legs remain, resumable"; exit 3; }; done
nl_say "all legs done"; exit 0
