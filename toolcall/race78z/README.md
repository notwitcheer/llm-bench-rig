# race78z: the same Hermes Agent task on two files of Qwen3.8-27B, side by side

queue item `85-nvfp4-race.sh` (staged 2026-11-03, enqueue is the operator's call), the race leg of item 78 (`../76-toolcall-nvfp4.sh`, ran as 78: NVFP4 on vLLM and UD-Q4_K_M on llama.cpp tie on tool calls, 85 vs 80 of 100, p=0.125; decode 56 vs 78 tok/s). capsule path `~/voxelbench/race78z/`.

one real Hermes Agent session per take (own `HERMES_HOME`, never the machine's always-on gateway), one server holding one file at a time (the two do not fit one 32 GB card together), the same prompt (`prompts/voxel-cottage-v1.txt`, an animated isometric voxel cottage on a canvas page), identical sampling sent through the provider's `extra_body`. two takes per file; the clip uses take 1 of each unless a recorder fault. the two takes are composited with the start synced on each take's own Enter mark; the clip says so.

| file | what it does |
|---|---|
| `85-nvfp4-race.sh` | the queue item: plan, drain, vLLM + tool-call smoke + takes, llama.cpp + smoke + takes, restore, gates, clip, summary |
| `r78z_take.sh` | one filmed take against a running server (proxy, private home, recorder, copy-out) |
| `r78z_mkhome.py` | the take's `HERMES_HOME` (named provider + `extra_body` sampling, `installs/` + `tools/` symlinked from the main root) |
| `r78z_proxy.py` | counting proxy + 1 Hz sampler; `srv_tokens` = the server's own live token count on both engines (llama: finished-request `predicted_n` + busy-slot `n_decoded`; vLLM: `vllm:generation_tokens_total`) |
| `r78z_record.py` | pty recorder (pyte), verbatim copy of item 80's `four_record.py` |
| `r78z_gate.py` | CPU liveness gate for the page, verbatim copy of item 80's `four_gate.py` |
| `r78z_render.py` | `race`: two panes + live HUD, real speed when the longer take is <= 240 s, else a labelled sped-up middle; `card`: the end card |
| `r78z_results.py` | the two finished pages side by side, then concatenates race + results + card |
| `r78z_facts.py` | the end card's numbers, read from the item-78 result files (exits 2 if they are not 85 vs 80 and 56 vs 78) |
| `r78z_summary.py` | `summary.json`, written last |
| `r78z_mock.py`, `test-78z-stub-e2e.sh` | CPU proof: engine-shaped mocks (vLLM-flavoured 56 tok/s, llama-flavoured 78 tok/s) drive real Hermes sessions through the whole item, ~6 min, no card |

derived from `voxelbench/four-agents/` (item 80). traps and numbers: vault `99-personal/benchmark-rig/toolcall-nvfp4-status.md`, section "78z race leg".
