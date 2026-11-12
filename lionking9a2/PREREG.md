# PREREG: 9a2 lion-king-neurons (causal neuron circuits, arXiv 2611.13885)

Frozen before any counted output. sha256 in FROZEN.sha256; the item's PLAN phase refuses to run on a hash mismatch.

## model and code
- model: Qwen/Qwen3-1.7B-Base (the paper's model; rev ea980cb0a6c2ae4b936e82123acc929f1cec04c1; downloaded 2026-11-12, 3,441,185,608 bytes model.safetensors, size-checked vs the HF API). amended 2026-11-12 from the cached instruct checkpoint (Qwen/Qwen3-1.7B, rev 70d244cc), which fails the margin gate (lk m0 0.125 < 3 x noise 0.250). the pre-amendment file is PREREG.v1-instruct.md (sha256 f3faa4e7...).
- repo: github.com/omtekyav/causal-neuron-circuits, sha d09f034f1e6a268a231dbbb7420217a8063687be (MIT).
- the repo's own inclusion gate (run_chains.py:87): a chain is included only if its comp1 margin >= min_margin (1.0) and the greedy answer is correct. the gate is NOT changed here.

## question
does the Lion King chain (answer Scar, bridge Simba) flip when a 12-neuron group from the Lion King prompt is patched, and is the flipping group unique?

## arms (the repo's own, unchanged)
- P_hop1: patch the 12 neurons that carry hop 1 (the paper's headline arm).
- P_random: 12 layer-matched random neurons (control).
- Z_zero: zero-ablation of the 12 neurons on the wrong chain (reference).
- subsets: all 4,096 subsets of the 12 neurons per arm (the repo enumerates these).

## flip rule (matches src/score.py: flipped = margin < 0)
- margin = logit(correct) - logit(wrong) at the last position; m0 = unpatched margin.
- a subset flips when its patched margin < 0. the summariser reads the saved P_hop1 delta and flips a subset when m0 - delta < 0.
- a flip is evidence only when m0 >= the noise floor: noise = max |Δ_repeat - Δ| over repeated bf16 runs (run_chains.py noise block). a chain whose m0 is under 3 x noise is INVALID-NOISE, regardless of the flip.
- minimal flipping set = a flipping subset with no strictly smaller flipping subset.
- not unique = two minimal flipping sets that are disjoint (share no neuron).

## verdict words (applied by summarize.py, nothing is worded by hand)
- INVALID-NOISE: m0 < 3 x noise_P_hop1. no claim.
- REPRODUCED ON ONE CHAIN: m0 >= 3 x noise, P_hop1 flips the full 12-neuron patch, P_random does NOT flip it, and the greedy patched answer starts with the bridge (Simba).
- PARTIAL: P_hop1 flips the full patch but one of the other two conditions fails.
- NOT REPRODUCED: P_hop1 does not flip the full patch.

## line 1 (built by the rule, not by hand)
- REPRODUCED ON ONE CHAIN: "the lion king's answer flips when we patch 12 neurons from another question, and [N] of 4,096 subsets flip it. [unique | not unique: D disjoint pairs]. one chain on qwen3-1.7B-base, the paper's model"
- other verdicts: a plain statement of the flip count and the deviation.

## extension (labelled, post hoc, never the headline)
- all 20 chains with the repo's gate forced OFF (SMOKE_FORCE / forced.json lists every chain), so the margin split can be read. reported separately; the 1.0 gate is the repo's and is not changed.
- the margin split (m0 < 4 vs >= 4) is the paper's exploratory cut, reported post hoc.

## amendment log
- 2026-11-12 (pre-count): model id changed Qwen3-1.7B -> Qwen3-1.7B-Base (see model and code). line-1 wording aligned to summarize.py ("from another question", the activations come from the hop1 question). both edits made after the forced-inclusion smoke on Base (smoke-only, never counted) and before any counted run. the counted run uses configs/chains-card5090-base.yaml, a copy of chains-card5090.yaml with model Qwen/Qwen3-1.7B-Base.
- frozen files (FROZEN.sha256, paths relative to ~/esoteric/lionking9a2/): PREREG.md, summarize.py, render_lk.py, configs/chains-card5090-base.yaml (copied into the repo configs/ at freeze). the repo's own run_chains.py and run_chains_probe.py are pinned by the repo sha d09f034 (probe: extension only, repo file untouched). the item script is at ~/voxel-stage/9a2-lion-king-neurons.sh and is md5-logged in the status note.

## what this does NOT claim
- not a reproduction of the paper's full numbers: one chain (lk), one model, one bf16 run per arm.
- not the Goodfire example itself (the post is cited, not reproduced).
- the 5090 timing is UNVERIFIED until a real run.
- the cached instruct checkpoint result (INVALID-NOISE on lk) is kept as the first finding, not the headline.
