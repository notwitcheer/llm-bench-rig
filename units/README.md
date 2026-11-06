# units: unit conversion inside tool arguments

Status: run 2026-11-06 01:10-01:29 CET on one RTX 5090 (800 calls + 400 bridge rows). Pre-registered verdict: SLIPS FOUND, 40 power-of-ten slips in 800 calls (Qwen3.8-27B Q6_K 2, Ornith-1.5-35B Q4_K_M 7, gpt-oss-20b MXFP4 14, Qwen3.5-9B Q4_K_M 8, Q8_0 9). 37 of the 40 are a millisecond figure copied unchanged into a seconds argument; 0 slips on the canonical and larger-unit forms. Qwen3.5-9B Q4 vs Q8: exact McNemar p = 0.45, no difference shown. Bridge parity with the paper's per-item answers: 396 of 400. Aggregate and readout: `data/units-orbit/`.

arXiv 2609.25009 tested numeric representation invariance in model answers. This folder tests the axis that paper does not: the user states a quantity in one unit, a tool argument is defined in another (the schema names the unit), and the model must convert. A power-of-ten slip there is a wrong action.

- `PREREG.md`: design, scoring, verdict words and limits, written before any model output.
- `tasks_def.py` -> `freeze.py` -> `tasks.json`, `FROZEN.sha256` (task set), `CODE.sha256` (harness). 40 tasks x 4 forms. 24 tasks use real Hermes tool schemas, extracted verbatim from a pinned checkout (`extract_schemas.py`, `schemas.json`, file + line recorded); 16 use synthetic tools (cents, metres, milliseconds, grams), labelled synthetic.
- `run_units.py`: single-turn `/v1/chat/completions` driver (llama-server `--jinja`), greedy, resumable.
- `score.py`: exact-rational classifier (ok / magnitude_slip / other_value / no_call / wrong_tool / malformed), per-model verdict, paired McNemar for the Qwen3.5-9B Q4/Q8 pair, line 1 built from the data.
- `mock_units.py`, `aggregate_selftest.py`, `make_mock_results.py`, `selftest_render.sh`: CPU proofs on scripted answers (not model output).
- `render_orbit.py`: the 16:9 orbit-ring video (numpy + PIL + ffmpeg, CPU).
- `bridge.py`: optional runtime-parity check against the paper's per-item answers (needs the paper's ancillary archive, not included).
- `queue/88-units-orbit.sh`: the idle-queue item (capsule paths).

Subject: how models convert units into tool arguments. Nothing here says anything about how Hermes behaves.

- post-run render fix (2026-11-06, CPU only, no new numbers): the running "slips so far" total no longer overlaps the last prompt form; CODE.sha256 updated for render_orbit.py. the run itself used the earlier hash.
