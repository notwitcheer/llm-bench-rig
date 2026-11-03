# reference/flock (paper-to-demo episode 2, arXiv 2608.10844)

CPU reference for the pre-registered gate of item 83 (`queue/83-paper-ep2-hovm.sh`, prompt `prompts/hovm-v1.txt`). Kept here because the gate's thresholds are justified by it.

- `hovm_ref.html`: the reference page (passes both gates). Written from the prompt's spec, not a model output. `MODE` is replaced by sed for the red-path stubs (`tests/test-flock-gates.sh`).
- `vicp.c`: independent C simulation, `gcc -O3 -march=native -o vicp vicp.c -lm`. `vicp N rho alpha seed eta start steps window [mode]`; mode 0 = mutual-triangle triplets (the authors' CPU heat-map code), mode 1 = the paper TEXT (bird i inside its own neighbour set, v0-normalised), mode 2 = star pairs among the neighbours EXCLUDING i (the authors' GPU code `FSE.cu`, the rule the prompt specifies). env `TRACE=<n>` prints the order parameter every n steps, `SCHED="eta:steps,eta:steps"` runs a noise schedule.
- `runsweep.sh`: xargs wrapper (capsule, niced, 8 procs). `results-*.txt`: the raw sweeps (random start 10^4 steps, mean of the last 200; `ord_g` = ordered start 1500 steps, mean of the last 300; `variants-text-rule` = the 2026-11-02 sweep that followed the paper text).
- `flock_core_trianglerule.js`: node-side port of mode 0, kept for the record (not used by the gates).
- the authors' data and code (CC BY 4.0): Zenodo 10.5281/zenodo.20793454 (`Finalized_Figs.zip`, fetched 2026-11-03), not copied here.
