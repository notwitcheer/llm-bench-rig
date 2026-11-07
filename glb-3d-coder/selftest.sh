#!/bin/bash
# CPU proof of the whole chain on a mock endpoint: generation driver (resume), checker (pass + each fail branch), scorer branches, renderer. prints one OK line on success.
set -uo pipefail
D=/home/witcheer/glb-3d-coder; PY=/home/witcheer/pw-env/bin/python; PYX=/home/witcheer/benchmark-rig/.venv/bin/python
T=${1:-/home/witcheer/scratch/glb-selftest}; PORT=${2:-8653}; RENDER=${GLB_SELFTEST_RENDER:-1}
rm -rf "$T"; mkdir -p "$T/res" "$T/out"
python3 $D/mock_glb.py $PORT glb4b-q8=good qwen38-27b-q4km=mixed ornith15-35b-q4km=mixed > "$T/mock.log" 2>&1 &
MP=$!; trap 'kill $MP 2>/dev/null' EXIT; sleep 1
for K in glb4b-q8 qwen38-27b-q4km ornith15-35b-q4km; do
  python3 $D/run_glb.py --model $K --port $PORT --out "$T/res" --prompts lighthouse-keeper,teapot-tower,pancake-ship,toaster-knight,snail-bakery,duck-submarine,haunted-vending --seeds 20261105 --inflight 3 > "$T/gen-$K.log" 2>&1 || { echo "FAIL gen $K"; tail -3 "$T/gen-$K.log"; exit 1; }
done
# resume: a second call must do nothing
RS=$(python3 $D/run_glb.py --model glb4b-q8 --port $PORT --out "$T/res" --prompts lighthouse-keeper,teapot-tower,pancake-ship,toaster-knight,snail-bakery,duck-submarine,haunted-vending --seeds 20261105 2>&1)
[[ $RS == *"0 cells to do"* ]] || { echo "FAIL resume: $RS"; exit 1; }
$PYX $D/check_cells.py --res "$T/res" --workers 3 --netcache /home/witcheer/glb-3d-coder/netcache > "$T/check.log" 2>&1 || { echo "FAIL check"; tail -5 "$T/check.log"; exit 1; }
N=$(grep -c . "$T/res/checks.jsonl"); [ "$N" = 21 ] || { echo "FAIL checks rows $N != 21"; exit 1; }
python3 - "$T/res/checks.jsonl" <<'E' || exit 1
import json,sys,collections
rows=[json.loads(l) for l in open(sys.argv[1])]
good=[r for r in rows if r["model"]=="glb4b-q8"]
assert all(r["full_pass"] for r in good), [r["reason"] for r in good if not r["full_pass"]]
reasons=collections.Counter(r["reason"].split(":")[0] if r["reason"] else "pass" for r in rows if r["model"]!="glb4b-q8")
need={"no page extracted","no export control","page error","empty scene","glb invalid"}
print("branches seen:", dict(reasons))
missing=need-set(reasons); 
if missing: print("FAIL branches not hit:", missing); sys.exit(1)
E
# scorer branches on fake cell counts
python3 $D/score.py --res "$T/res" --out "$T/out" --cells 7 > "$T/score.log" 2>&1 || { echo "FAIL score"; cat "$T/score.log"; exit 1; }
echo "score(mock,7 cells): $(cat $T/score.log)"
SO=$(python3 $D/score.py --res "$T/res" --out "$T/out" --cells 24); [[ $SO == PROVISIONAL* ]] || { echo "FAIL provisional branch: $SO"; exit 1; }
# BROKEN branch: a 4B with no valid page
mkdir -p "$T/res2"; cp "$T/res/checks.jsonl" "$T/res2/"; for K in glb4b-q8 qwen38-27b-q4km ornith15-35b-q4km; do cp "$T/res/$K.jsonl" "$T/res2/"; done
python3 - "$T/res2/checks.jsonl" <<'E'
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1])]
for r in rows:
    if r["model"]=="glb4b-q8": r.update(page_valid=False,full_pass=False,glb_valid=False,nonempty=False)
open(sys.argv[1],"w").write("".join(json.dumps(r)+"\n" for r in rows))
E
SO=$(python3 $D/score.py --res "$T/res2" --out "$T/out" --cells 7); [[ $SO == BROKEN* ]] || { echo "FAIL broken branch: $SO"; exit 1; }
python3 - "$T/res2/checks.jsonl" <<'E'
import json,sys
rows=[json.loads(l) for l in open(sys.argv[1])]
for r in rows:
    if r["model"]=="glb4b-q8": r.update(page_valid=True,full_pass=True,glb_valid=True,nonempty=True)
    if r["model"]!="glb4b-q8": r.update(full_pass=False)
open(sys.argv[1],"w").write("".join(json.dumps(r)+"\n" for r in rows))
E
SO=$(python3 $D/score.py --res "$T/res2" --out "$T/out" --cells 7); [[ $SO == "AHEAD OF BOTH"* ]] || { echo "FAIL ahead branch: $SO"; exit 1; }
echo "verdict branches ok: PROVISIONAL, BROKEN, AHEAD OF BOTH"
if [ "$RENDER" = 1 ]; then
  nice $PYX $D/render_glb.py --res "$T/res" --out "$T/render" --mock --prompts lighthouse-keeper,teapot-tower --secs 1 > "$T/render.log" 2>&1 || { echo "FAIL render"; tail -5 "$T/render.log"; exit 1; }
  ls "$T/render"/*.mp4 "$T/render"/contact-*.png > /dev/null || { echo "FAIL render outputs"; exit 1; }
  echo "render ok: $(ls $T/render | tr '\n' ' ')"
fi
echo "SELFTEST OK"
