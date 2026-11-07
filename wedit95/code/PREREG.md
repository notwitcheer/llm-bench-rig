# PREREG: a local Qwen3.8-27B in Hermes Agent as the weight editor of Qwen3.5-0.8B (idle-queue item 95-weight-editor)

Frozen 2026-11-06 (CET), before any counted output. The files listed in `FROZEN.sha256` (this file, the arms, the verdict rule, the tool service, the editor's task prompt, the published edit) must hash to what is recorded: the item's PLAN phase refuses to run if any differ. Smoke runs use the slug `wedit95-smoke` and are never counted.

## the question

Weight-space programming (MIT CSAIL, 2026-11-02, https://rikiyat.github.io/blog/weight-space-programming/): a rank-limited edit of two MLP matrices of Qwen3.5-0.8B (layers 15 and 19 `mlp.down_proj`, 64 neurons each, https://huggingface.co/RikiyaT/Qwen3.5-0.8B-weight-space-programming) cures the model's thinking loops, found in the paper by a frontier-model editor. Home replications read 2026-11-06: a scripted pipeline with no language model reports thinking closed on 15/18 problems against 2/18 for the base and a bit-exact rebuild of the published edit 17/18 (https://github.com/KidIkaros/wsp-reproduction); a blind frontier-editor repair search in another replay went 5/64 to 3/64 (https://github.com/jsaurabh/weight-space-lab). So the edit effect is not the open question; the open question is whether a LOCAL editor finds an edit that works.

**Can Qwen3.8-27B (Q6_K, llama.cpp), driving a real Hermes Agent session with a sandboxed tool for forward passes and candidate edits of Qwen3.5-0.8B, find an edit that lowers the 0.8B model's thinking-loop rate on held-out problems, and does it match the scripted pipeline?** "The local editor could not match the scripted pipeline" is a valid, pre-registered result, and so is "neither cured it". The subject of any failure is the editor model and the task, never the agent harness.

## arms (all on the same base GGUF: Qwen/Qwen3.5-0.8B rev 2fc06364715b967f1860aea9cf38778875588b17 converted to bf16 GGUF, edits patch the bf16 bytes of `blk.L.ffn_down.weight`, byte-equality with the safetensors tensor checked at staging)

- **base**: unedited.
- **published**: the MIT CSAIL p50c edit applied bit-exact (every rebuilt tensor's sha256 equals the manifest's `edited_sha256`, else the arm does not exist). Reference arm: if it does not reproduce on our rig, the verdict is EDIT DID NOT REPRODUCE and no editor comparison is made.
- **scripted** (arm B): `wd_scripted.py`, no language model: written from the stage-2 diagnosis of KidIkaros/wsp-reproduction (`src/driver2.py`): d_act = mean MLP intermediate over looping contexts minus over productive contexts, d = normalised (W_down @ d_act), neurons = top 64 by |d_act|, strength 6, at layers 15 and 19. The layers are the replication's (taken from the paper's released edit); the editor is NOT told them. Evidence = the same saved dev-set base thinking texts the editor can read (`wd_acts.pick_rows`: first 6 looping tails, first 4 heads).
- **editor** (arm A): Qwen3.8-27B Q6_K (22.9 GB, KV q8_0, ctx 65,536, llama-server of the Donald build 839a572a4) driving ONE Hermes Agent session (pin 9bda188, own HERMES_HOME, minted launcher, toolsets terminal + file, approvals off, memory off, max 90 turns, thinking on, temp 0.6 / top-p 0.95 / top-k 20 / seed 721, never the main gateway, never :8090). Its only access to the 0.8B model is the `wd` command (sandbox service `wd_service.py`): `info`, `transcripts` (saved base thinking texts), `acts` (a forward pass over a text it wrote: mean MLP intermediate per layer), `weights L` (base down_proj), `apply SPEC` (validated candidate GGUF), `eval NAME` (dev set), `result NAME`, `submit NAME`. Limits per candidate: <= 2 layers, <= 128 neurons per layer, |strength| <= 12, rule W[:, j] <- W[:, j] - s (W[:, j] . d) d. Budget: 6 dev evaluations, ~55 minutes of tool time, wall cap 60 minutes. The editor is not told the layer numbers, neuron counts or strength of the published edit, nor that one exists; its task text is `prompts/edit-task.txt` (frozen). Its python is a numpy-only venv; no network is configured for it.
- the editor's final candidate = the BEST BY DEV loop rate among those it had scored (ties: the earlier evaluation). `wd submit` is not used for selection. A take that scored no candidate is NO CANDIDATE.

## problem sets and metric (GSM8K, openai/gsm8k cached, `random.Random(20261106)`)

- DEV (the editor's own measuring stick, also used to build the scripted evidence): 20 problems from the TRAIN split. HELD-OUT: 40 problems from the TEST split, never reachable from the tool, never in any evidence. One try per problem and arm, the same per-problem seeds (1000 + i) for every arm. Thinking on, `chat_template_kwargs {enable_thinking: true}`, temperature 0.6, top-p 0.95, top-k 20, presence penalty 1.5 (the replication's sampler), max_tokens 16,384.
- **loop** = the try's answer field is empty at the cap, i.e. the thinking never closed. **loop rate** = loops / tries. Secondary, reported only: answered correctly (final `\boxed{}` equals the GSM8K gold), mean completion tokens.

## verdicts (applied by `wd_summary.py` on the held-out counts, in this order; nothing worded by hand; `--selftest` proves every branch)

"cured(arm)" = one-sided exact Fisher test that the arm's loop count is LOWER than the base's, p < 0.05.
1. INCOMPLETE: an arm or the editor take is missing -> exit 3, resumes, no summary.json.
2. INVALID: the base's held-out loop rate is under 40% (the failure did not reproduce on this rig; nothing to cure).
3. EDIT DID NOT REPRODUCE: the published edit is not cured() -> the setup, not the editors, is the finding.
4. NO CANDIDATE: the editor scored no candidate in its time box.
5. EDITOR BEAT THE SCRIPT / EDITOR MATCHED THE SCRIPT / EDITOR HELPED, SCRIPT BETTER: both cured; the editor's loop count is significantly lower / no significant difference / significantly higher than the script's (Fisher, p < 0.05).
6. EDITOR CURED, SCRIPT DID NOT: only the editor cured. EDITOR COULD NOT MATCH THE SCRIPT: only the script cured. NEITHER CURED IT: neither did.
- line 1 is built by `line1()` from the verdict and the loop counts. Flags the gate sets on the take (`outside-workdir`: the agent's tool calls name the harness, the arms or the published edit; `network`; `no-wd-use`) are printed with line 1, and a flagged take is never presented as a clean result.

## honest limits to print with any number

one 0.8B model; GSM8K word problems; 40 held-out tries per arm (a loop-rate difference under ~20 points is not detectable); ONE editing session of ONE local 27B (this is not a benchmark of Hermes Agent or of Qwen3.8-27B as an editor); the scripted arm is our reading of the replication's pipeline with its layers given, not the authors' code, and not a claim about their result; the published edit's discovery and the effect are the MIT authors'; 16,384-token cap (the paper's cap was 32,768); presence penalty 1.5 and the sampler are the replication's, not necessarily the paper's; bf16 GGUF on llama.cpp, not vLLM; the editor chose its own layers, which may differ from the paper's.

## render

CPU only on capsule (pyte + PIL + the static ffmpeg), 1920x1080, ~64 s: result card first (8 s), then three panes (the unedited model's looping thinking text, the editor's real Hermes pane time-compressed with a speed pill, the edited model's thinking on the same problem) with the loop-rate counters and the editor's dev scores arriving as the take reaches them, then the card again. Mock data is stamped MOCK on every frame; `render_wedit.py` is not frozen.
