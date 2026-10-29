# final-gate (trees episode)

`gate_trees.py` is the FINAL, pre-registered gate of the trees episode. It is deliberately NOT in `scripts/`:
on capsule it is deployed to `~/.vxfinal/gate_trees.py` (dot directory, mode 700) so the agent that is being
graded has no obvious path to the criteria (HARDENING L33). The in-run gate the agent sees is
`scripts/quick_gate_trees.py`, which carries no paper criteria.

- deploy: `ssh capsule-w 'mkdir -p ~/.vxfinal && chmod 700 ~/.vxfinal'; scp final-gate/gate_trees.py capsule-w:~/.vxfinal/gate_trees.py`
- the queue items 72c/72d/72e abort in their PLAN if the criteria text is visible under `~/voxelbench` or `~/idle-queue/*.sh`
- tests (CPU, SwiftShader): `bash tests/test-trees-gates.sh` (stub matrix) and `python3 tests/test_vx_stall.py`
- the h2 original (md5 30286d6ce8eba8f38eaaeb087eba6733, `scripts/gate_trees.py` in commit d0242d3 and earlier) has the same pass rule; h3 adds `main_thread_ok`, `main_thread` and reported-only `growth_flags`
