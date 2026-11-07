#!/bin/bash
# CPU proof clone of 97-arete-msm.sh (no card, no Donald, no :8090). differs from the item ONLY by the sed lines below + the SwiftShader copies of the browser scripts.
# the tiny probe model stands in for both arms (CUDA hidden, port 8199, WALL=4 TURNS=2); the PROOF-HOOK lines copy a stub page into work/ for takes 1q 2q 2a and leave 1a empty (the fail path).
set -u
S=$HOME/voxel-stage/arete97; SRC=$S/97-arete-msm.sh; CL=$S/proof/clone-97.sh; PS=$S/proof/scripts
TINY=$HOME/models/tiny-probe/qwen2.5-0.5b-instruct-q4_k_m.gguf
$HOME/benchmark-rig/.venv/bin/python - <<'PYEOF'
import re, os
ps = os.path.expanduser("~/voxel-stage/arete97/proof/scripts")
for f in ("gate.py", "orbit_clip.py", "orbit_clip_multidrag.py"):
    p = f"{ps}/{f}"; s = open(p).read()
    s = re.sub(r'^ARGS = \["--use-angle=gl-egl", "--ignore-gpu-blocklist"\].*$', 'ARGS = ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader", "--ignore-gpu-blocklist"]  # CPU PROOF COPY', s, flags=re.M)
    assert "swiftshader" in s, f; open(p, "w").write(s)
PYEOF
HOOK='case "$SLUG" in *t1-arete*) : ;; *) cp '$S'/stub/stub.html "$V/runs/$SLUG/work/index.html";; esac   # PROOF-HOOK'
sed -e 's#^NL_TAG=.*#NL_TAG=vox97proof#' \
    -e 's#^MANIFEST=/home/witcheer/voxel-vox97arete-status.json#MANIFEST=/tmp/a97-proof-manifest.json#' \
    -e 's#^SLUGPFX=.*#SLUGPFX=a97proof#' \
    -e "s#^QFILE=.*#QFILE=$TINY#" -e "s#^AFILE=.*#AFILE=$TINY#" \
    -e 's#^QSHA=[0-9a-f]*#QSHA=74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db#' -e 's#^ASHA=[0-9a-f]*#ASHA=74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db#' \
    -e "s#^QBYTES=.*#QBYTES=$(stat -c %s $TINY)#" -e "s#^ABYTES=.*#ABYTES=$(stat -c %s $TINY)#" \
    -e 's#^PORT=8090; export PORT.*#PORT=8199; export PORT; export CUDA_VISIBLE_DEVICES=""#' \
    -e 's#^CTX=131072; WALL=60; TURNS=20; KV=q8_0#CTX=131072; WALL=4; TURNS=2; KV=q8_0#' \
    -e 's#^VS_SECS=30; VS_PRESET=slow#VS_SECS=8; VS_PRESET=ultrafast#' \
    -e "s#^GATE=.*#GATE=$PS/gate.py#" -e "s#^QG=.*#QG=$PS/quick_gate_msm.py#" \
    -e 's#\$V/scripts/orbit_pick.py#'$PS'/orbit_pick.py#g' \
    -e 's#"\$V/scripts/\$OSC"#"'$PS'/$OSC"#' \
    -e 's#\$S/run#$S/proof/run#g' -e 's#^VSDIR=.*#VSDIR=$V/runs/vs-a97proof#' \
    -e 's#^\. /home/witcheer/night-lib.sh#. /home/witcheer/night-lib.sh; nl_donald_stop() { :; }; nl_donald_restore() { :; }#' \
    -e 's#^DRAINED=0; PAUSED=0; FAILED=0; first=1#mem_used() { echo 0; }; donald_back() { nl_say "  (clone) no Donald restore"; return 0; }; rule3_check() { nl_say "  (clone) rule 3 skipped"; return 0; }\nDRAINED=0; PAUSED=0; FAILED=0; first=1#' \
    -e 's#--clip-seconds 30#--clip-seconds 30 --swiftshader#' -e 's#--seconds 30 --orbit-px "\$OPX" --keys "n@11,n@22"#--seconds 12 --orbit-px "$OPX" --keys "n@4,n@8"#' \
    -e 's#^  HTML=\$V/runs/\$SLUG/work/index.html$#  '"$(printf '%s' "$HOOK" | sed 's/[&#\\]/\\&/g')"'\n  HTML=$V/runs/$SLUG/work/index.html#' \
    $SRC > $CL
echo "--- diff vs the item (clone lines only):"; diff <(cat $SRC) $CL | grep '^>' | cut -c1-230
grep -n '8090' $CL | grep -v '^.*#' | head; grep -c 8090 $CL
bash -n $CL && echo clone-syntax-ok
rm -f /tmp/a97-proof-manifest.json
rm -rf $S/proof/run $HOME/voxelbench/runs/vs-a97proof
mkdir -p $S/proof/run
