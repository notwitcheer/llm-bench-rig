#!/bin/bash
# 70-hrl-full250.sh: idle-queue item (2026-10-05 ~22:35 CEST, mercury, operator "following your lead ... so it works
# well with the GPU runs already scheduled"). queued BEHIND 09v (duck) and 59 (matharena) by lexical order on purpose.
# hermes-harness transfer check, full SmolDataEnvs test split (250 tasks) x 4 LFM2.5-2.6B checkpoints, all inside
# Hermes Agent v2026.9.24 (baked image hrl/env-data-agent-hermes:2026.9.24, patched Harbor adapter, OpenEnv hermes
# seam re-added locally). protocol = FineEnvs eval (temp 0.8, 4096 tokens, 17 steps, 600 s agent timeout, numeric grader).
# pace measured 2026-10-05: 25 tasks in 8m21s at C=6 -> ~84 min per 250-task leg + ~1 min server start/stop.
# one leg = one checkpoint; a leg starts only with >= 100 min left in the block, else rc=3 (resumable, stays queued).
# leg artifact = runs/full-<tag>-n250/summary.json (fresh). a leg dir without it is moved aside before a retry.
# Donald: hrl-probe.sh drains and restores per leg; the trap below is a second net.
# status note: kept in the private ops log (hermes-harness-transfer-status.md)
MANIFEST=/home/witcheer/hrl-full250-status.json
export NL_TAG=hrl250
. /home/witcheer/night-lib.sh
NL_FAIL=0
H=/home/witcheer/harness-rl
LEGS="base mhsft mhrl ocrl"
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
  out="$H/runs/full-$leg-n250"
  art="$out/summary.json"
  if nl_skip_exact "$art"; then nl_say "leg $leg already done ($art)"; continue; fi
  left=$(( DEADLINE - $(date +%s) ))
  if [ "$left" -lt 6000 ]; then nl_say "leg $leg needs ~90 min, only $((left/60)) min left in the block: pausing"; remaining=1; break; fi
  [ -d "$out" ] && { mv "$out" "$out.incomplete-$(date +%m%d-%H%M)"; nl_say "moved an incomplete $leg dir aside"; }
  nl_mark "$leg" running
  t0=$(date +%s)
  N=250 C=6 TAG="full-$leg" CKPT="$(ckpt_of $leg)" bash "$H/hrl-probe.sh"
  rc=$?
  nl_judge_artifact "$leg" "$rc" "$art" "$t0" "$(( ($(date +%s)-t0)/60 )) min" || { nl_say "leg $leg FAILED rc=$rc"; exit 1; }
  nl_say "leg $leg done in $(( ($(date +%s)-t0)/60 )) min"
done
for leg in $LEGS; do nl_skip_exact "$H/runs/full-$leg-n250/summary.json" || remaining=1; done
[ "$remaining" = 1 ] && exit 3
nl_say "all four legs done"
exit 0
