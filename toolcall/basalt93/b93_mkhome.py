#!/usr/bin/env python3
"""b93_mkhome.py <root> <tag> <proxy-port> <model-alias> <ctx> [--think on|off]

Makes ONE take's private HERMES_HOME under <root>/home-<tag>/ (a directory of its own, NOT a profile of the main install: the main hermes-gateway.service never serves it).
Same recipe as the item-80 four_mkhome.py (item 89 copy) (main root's installs/ and tools/ symlinked in; launch with HERMES_DISABLE_LAZY_INSTALLS=1 and the minted launcher), with two
changes for the race between two FILES:
* a named custom provider with `extra_body`, so BOTH takes send the identical sampling fields (temperature 0.6, top_p 0.95, min_p 0.05, seed 721: ours, not a vendor setting;
  the model card's own sampling keys differ by engine default, so the fields are pinned) so every take sends the identical fields. The mock logs what Hermes really sends (MOCK_LOG).
* `--think off` adds chat_template_kwargs.enable_thinking=false (default is the template default = thinking on, the regime item 78 scored).
"""
import os
import sys

root, tag, port, alias, ctx = sys.argv[1:6]
think = sys.argv[sys.argv.index("--think") + 1] if "--think" in sys.argv else "on"
home = os.path.join(root, "home-" + tag)
os.makedirs(home, exist_ok=True)
os.chmod(home, 0o700)
kw = "" if think == "on" else "      chat_template_kwargs:\n        enable_thinking: false\n"
cfg = """model:
  provider: fnlocal
  default: %(alias)s
  context_length: %(ctx)s
providers:
  fnlocal:
    api: http://127.0.0.1:%(port)s/v1
    default_model: %(alias)s
    api_key: local-key
    extra_body:
      temperature: 0.6
      top_p: 0.95
      min_p: 0.05
      top_k: 20
      seed: %(seed)s
%(kw)stoolsets:
  - terminal
  - file
agent:
  max_turns: 60
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
""" % {"alias": alias, "port": port, "ctx": ctx, "kw": kw, "seed": os.environ.get("RP_SEED", "721")}
open(os.path.join(home, "config.yaml"), "w").write(cfg)
main = os.environ.get("FR_MAIN_HOME", os.path.expanduser("~/.hermes"))
for sub in ("installs", "tools"):
    dst = os.path.join(home, sub)
    if not os.path.lexists(dst):
        os.symlink(os.path.join(main, sub), dst)
open(os.path.join(home, "SOUL.md"), "w").write("You are a capable autonomous coding agent. Work without asking the user questions: nobody is watching this session.\n")
print(home)
