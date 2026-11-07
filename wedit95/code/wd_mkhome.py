#!/usr/bin/env python3
"""wd_mkhome.py <root> <tag> <proxy-port> <model-alias> <ctx>: ONE take's private HERMES_HOME under <root>/home-<tag>/ (a directory of its own, never a profile of the main install; the always-on gateway never serves it).
Same recipe as item 89's rp_mkhome.py (main root's installs/ and tools/ symlinked in; launch with HERMES_DISABLE_LAZY_INSTALLS=1 and the minted launcher). Sampling is ours (Qwen's thinking-mode values + a fixed seed), sent through extra_body."""
import os
import sys

root, tag, port, alias, ctx = sys.argv[1:6]
home = os.path.join(root, "home-" + tag)
os.makedirs(home, exist_ok=True)
os.chmod(home, 0o700)
cfg = """model:
  provider: editor
  default: %(alias)s
  context_length: %(ctx)s
providers:
  editor:
    api: http://127.0.0.1:%(port)s/v1
    default_model: %(alias)s
    api_key: local-key
    extra_body:
      temperature: 0.6
      top_p: 0.95
      top_k: 20
      min_p: 0.0
      seed: %(seed)s
toolsets:
  - terminal
  - file
agent:
  max_turns: 90
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
""" % {"alias": alias, "port": port, "ctx": ctx, "seed": os.environ.get("WD_SEED", "721")}
open(os.path.join(home, "config.yaml"), "w").write(cfg)
main = os.environ.get("FR_MAIN_HOME", os.path.expanduser("~/.hermes"))
for sub in ("installs", "tools"):
    dst = os.path.join(home, sub)
    if not os.path.lexists(dst):
        os.symlink(os.path.join(main, sub), dst)
open(os.path.join(home, "SOUL.md"), "w").write("You are a capable autonomous research engineer. Work without asking the user questions: nobody is watching this session.\n")
print(home)
