#!/usr/bin/env python3
"""four_mkhome.py <root> <agent> <proxy-port> <model-alias> <ctx> [--stream-reasoning]

Makes ONE agent's private HERMES_HOME under <root>/home-<agent>/ (a directory of its own, NOT a profile of the main install), so the
harness agents stay isolated from the machine's own always-on Hermes gateway and its profiles. Launch with
  HERMES_DISABLE_LAZY_INSTALLS=1 HERMES_HOME=<root>/home-<agent> <checkout>/.hermes/bin/hermes chat --cli --toolsets terminal,file
The home symlinks the main root's installs/ and tools/ (the dependency generation + store python): without them a new data root builds its own runtime
(python-olm cmake, minutes, then fails) and the checkout's old venv lacks modules (snowballstemmer), measured 2026-11-03.

The config is the series bench shape (run_agent.sh): custom provider on the agent's proxy port, explicit context_length (>= Hermes' 64,000
minimum), approvals off (nobody to click), memory off (no notes leaking between agents), telemetry off, toolsets terminal + file only (no
delegation: a sub-agent would add requests the race HUD cannot attribute), display compact + streaming + reasoning shown so the pane is alive.
"""
import os
import sys

root, ag, port, alias, ctx = sys.argv[1:6]
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
open(os.path.join(home, "config.yaml"), "w").write(cfg)
# a new data root would build its own dependency environment (python-olm cmake: minutes, then fails); share the main root's READ-ONLY runtime instead
main = os.environ.get("FR_MAIN_HOME", os.path.expanduser("~/.hermes"))
for sub in ("installs", "tools"):
    dst = os.path.join(home, sub)
    if not os.path.lexists(dst):
        os.symlink(os.path.join(main, sub), dst)
open(os.path.join(home, "SOUL.md"), "w").write("You are a capable autonomous coding agent. Work without asking the user questions: nobody is watching this session.\n")
print(home)
