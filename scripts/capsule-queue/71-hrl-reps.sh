#!/bin/bash
# 71-hrl-reps.sh: idle-queue item (2026-10-08 ~07:30 CEST, mercury, operator "yes for both" in the GPU thread).
# hermes-harness transfer check, REPEAT samples. the FineEnvs eval samples at temperature 0.8 with no seed, so each
# rerun of the same 250 held-out test tasks is an independent pass@1 sample. three more passes (r2..r4) x four
# LFM2.5-2.6B checkpoints -> mean pass@1 over 4 samples per task, to settle opencode-RL vs base (n=250 single pass:
# 22 gained / 13 lost, p 0.18). NOT the 1,000 "other" tasks: those are the training split the checkpoints learnt on.
# legs are pass-major (r2 all four, then r3, then r4) so a partial run stays balanced across checkpoints.
# same protocol, image, adapter patches and pace as 70-hrl-full250 (63-85 min per leg, ~1 min server start/stop).
# one leg = one checkpoint x one pass; starts only with >= 100 min left in the block, else rc=3 (resumable).
# leg artifact = runs/full-<ckpt>-r<k>-n250/summary.json. a leg dir without it is moved aside before a retry.
# Donald: hrl-probe.sh drains and restores per leg; the trap below is a second net.
# status note: kept in the private ops log (hermes-harness-transfer-status.md)
MANIFEST=/home/witcheer/hrl-reps-status.json
export NL_TAG=hrlreps
. /home/witcheer/night-lib.sh
NL_FAIL=0
H=/home/witcheer/harness-rl
LEGS="r2-base r2-mhsft r2-mhrl r2-ocrl r3-base r3-mhsft r3-mhrl r3-ocrl r4-base r4-mhsft r4-mhrl r4-ocrl"
ckpt_of() { case $1 in base) echo "";; mhsft) echo FineEnvs/LFM2.5-2.6B-multiharness-SFT;;
  mhrl) echo FineEnvs/LFM2.5-2.6B-multiharness-RL;; ocrl) echo FineEnvs/LFM2.5-2.6B-opencode-RL;; esac; }

# ---- preflight ----
nl_check_path "$H/hrl-probe.sh" "$H/FineEnvs/05-multi-harness-rl/.venv/bin/activate" "$H/FineEnvs/05-multi-harness-rl/eval/evaluate_hrl.py"
docker image inspect hrl/env-data-agent-hermes:2026.9.24 >/dev/null 2>&1 && nl_ok "baked image present" || nl_fail "baked image missing"
grep -q "FROM hrl/env-data-agent-hermes:2026.9.24" "$H/FineEnvs/05-multi-harness-rl/prepared/datasets/test/tasks/0000_324_324293_qa_2/environment/Dockerfile" && nl_ok "task Dockerfiles point at the bake" || nl_fail "task Dockerfiles not rewritten"
( cd "$H/FineEnvs/05-multi-harness-rl" && . .venv/bin/activate && python - <<'PY'
import harbor.agents.installed.hermes as m, openenv.harbor.seams as s, inspect
src = open(m.__file__).read()
assert "LOCAL PATCH hrl" in src and "ref = self._version" in src and "LOCAL PATCH hrl-skip" in src, "adapter patches missing"
assert any(isinstance(v, dict) and "hermes" in v for v in vars(s).values()), "hermes seam missing"
PY
) && nl_ok "adapter patches + hermes seam present" || nl_fail "adapter/seam patches missing"
nl_check_disk 60
nl_gate

# ---- run ----
trap nl_donald_restore EXIT
[ -f "$MANIFEST" ] || nl_manifest_init $LEGS
DEADLINE=$(nl_deadline_epoch)
remaining=0
for leg in $LEGS; do
  rep=${leg%%-*}; ck=${leg#*-}
  out="$H/runs/full-$ck-$rep-n250"
  art="$out/summary.json"
  if nl_skip_exact "$art"; then nl_say "leg $leg already done ($art)"; continue; fi
  left=$(( DEADLINE - $(date +%s) ))
  if [ "$left" -lt 6000 ]; then nl_say "leg $leg needs ~90 min, only $((left/60)) min left in the block: pausing"; remaining=1; break; fi
  [ -d "$out" ] && { mv "$out" "$out.incomplete-$(date +%m%d-%H%M)"; nl_say "moved an incomplete $leg dir aside"; }
  nl_mark "$leg" running
  t0=$(date +%s)
  N=250 C=6 TAG="full-$ck-$rep" CKPT="$(ckpt_of $ck)" bash "$H/hrl-probe.sh"
  rc=$?
  nl_judge_artifact "$leg" "$rc" "$art" "$t0" "$(( ($(date +%s)-t0)/60 )) min" || { nl_say "leg $leg FAILED rc=$rc"; exit 1; }
  nl_say "leg $leg done in $(( ($(date +%s)-t0)/60 )) min"
done
for leg in $LEGS; do rep=${leg%%-*}; ck=${leg#*-}; nl_skip_exact "$H/runs/full-$ck-$rep-n250/summary.json" || remaining=1; done
[ "$remaining" = 1 ] && exit 3
nl_say "all twelve legs done"
exit 0
