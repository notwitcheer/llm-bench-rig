#!/usr/bin/env python3
"""four_mkhome.py <root> <agent> <proxy-port> <model-alias> <ctx> [--stream-reasoning] [--compact-tokens N]

Makes ONE agent's private HERMES_HOME under <root>/home-<agent>/ (a directory of its own, NOT a profile of the main install), so the
harness agents stay isolated from the machine's own always-on Hermes gateway and its profiles. Launch with
  HERMES_DISABLE_LAZY_INSTALLS=1 HERMES_HOME=<root>/home-<agent> <checkout>/.hermes/bin/hermes chat --cli --toolsets terminal,file
The home symlinks the main root's installs/ and tools/ (the dependency generation + store python): without them a new data root builds its own runtime
(python-olm cmake, minutes, then fails) and the checkout's old venv lacks modules (snowballstemmer), measured 2026-11-03.

The config is the series bench shape (run_agent.sh): custom provider on the agent's proxy port, explicit context_length (>= Hermes' 64,000
minimum), approvals off (nobody to click), memory off (no notes leaking between agents), telemetry off, toolsets terminal + file only (no
delegation: a sub-agent would add requests the race HUD cannot attribute), display compact + streaming + reasoning shown so the pane is alive.

HOME CONFIG VERSIONS (item 80 and 80b share this file):
  v1 (default, item 80): no compression block. Hermes then compacts at max(0.50, 0.75 small-window floor) x context_length = 49,152 tokens for ctx 65536,
     so four agents can sit at 4 x ~49k = 196k, more than the 163,840-token pooled KV (measured 2026-11-03: all four requests rejected at about 39-41k each).
  v2-compact (80b): `--compact-tokens N` or env FR_COMPACT_TOKENS=N adds `compression.threshold_tokens: N`, the documented absolute cap on the compaction trigger
     (compaction fires at the LOWER of the ratio trigger and N), AND `compression.protect_first_n: 0`. context_length stays 65536 on purpose: Hermes refuses a window
     below 64,000, so the per-agent budget cannot be set through context_length. protect_first_n 0 matters as much as the cap: with the default 3 the first three rows
     (the task and the ~20k-token first write_file generation, reasoning included) are protected until the first compaction, and a compaction needs more than 7 messages
     (replayed 2026-11-03 on item 80's real transcript with the pinned compressor: 7 messages, default head = refused; head 0 = 21k -> 4k rough tokens at 5 messages).
     The version and N are written to <home>/home-version.txt so the run can prove which config each agent had.
"""
import os
import sys

root, ag, port, alias, ctx = sys.argv[1:6]
compact = os.environ.get("FR_COMPACT_TOKENS", "").strip()
if "--compact-tokens" in sys.argv:
    compact = sys.argv[sys.argv.index("--compact-tokens") + 1]
if compact and (not compact.isdigit() or int(compact) < 16000 or int(compact) >= int(ctx)):
    sys.exit("four_mkhome: --compact-tokens must be an integer in [16000, %s)" % ctx)
home = os.path.join(root, "home-" + ag)
os.makedirs(home, exist_ok=True)
os.chmod(home, 0o700)
cfg = """model:
  default: %(alias)s
  provider: custom
  base_url: http://127.0.0.1:%(port)s/v1
  api_key: local-key
  context_length: %(ctx)s
toolsets:
  - terminal
  - file
agent:
  max_turns: 80
  clarify_timeout: 20
approvals:
  mode: "off"
memory:
  memory_enabled: false
  user_profile_enabled: false
display:
  interface: cli
  compact: true
  streaming: true
  show_reasoning: true
telemetry:
  shared_metrics:
    enabled: false
    send: false
    offer_version: 2
""" % {"alias": alias, "port": port, "ctx": ctx}
if compact:
    cfg += "compression:\n  enabled: true\n  threshold_tokens: %s\n  protect_first_n: 0\n" % compact
open(os.path.join(home, "config.yaml"), "w").write(cfg)
open(os.path.join(home, "home-version.txt"), "w").write(("v2-compact threshold_tokens=%s protect_first_n=0 context_length=%s\n" % (compact, ctx)) if compact else ("v1 context_length=%s\n" % ctx))
# a new data root would build its own dependency environment (python-olm cmake: minutes, then fails); share the main root's READ-ONLY runtime instead
main = os.environ.get("FR_MAIN_HOME", os.path.expanduser("~/.hermes"))
for sub in ("installs", "tools"):
    dst = os.path.join(home, sub)
    if not os.path.lexists(dst):
        os.symlink(os.path.join(main, sub), dst)
open(os.path.join(home, "SOUL.md"), "w").write("You are a capable autonomous coding agent. Work without asking the user questions: nobody is watching this session.\n")
print(home)
