#!/usr/bin/env python3
"""chess_mkhome.py <arena> <agent> <port> <model-alias> <ctx> <goal-turns>

Makes ONE agent's private HERMES_HOME under <arena>/home-<agent>/ (its own data root, NOT a profile of the main install): the harness stays
isolated from the machine's always-on Hermes gateway and its profiles,
so the chess agents never touch that gateway. The home symlinks the main root's installs/ and tools/ (a new data root otherwise builds
its own runtime: python-olm cmake, minutes, then fails).

Config = the series bench shape (run_agent.sh): custom provider on the agent's OWN llama-server port (never :8090), explicit context_length, approvals off,
memory off, telemetry off, goals.max_turns, toolsets terminal + file only (no web search, no browser, no delegation). Prints the home path.
"""
import os
import sys

arena, ag, port, alias, ctx, turns = sys.argv[1:7]
home = os.path.join(arena, "home-" + ag)
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
  max_turns: 90
  clarify_timeout: 20
approvals:
  mode: "off"
memory:
  memory_enabled: false
  user_profile_enabled: false
goals:
  max_turns: %(turns)s
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
""" % {"alias": alias, "port": port, "ctx": ctx, "turns": turns}
open(os.path.join(home, "config.yaml"), "w").write(cfg)
main = os.environ.get("CHESS_MAIN_HOME", os.path.expanduser("~/.hermes"))
for sub in ("installs", "tools"):
    dst = os.path.join(home, sub)
    if not os.path.lexists(dst):
        os.symlink(os.path.join(main, sub), dst)
open(os.path.join(home, "SOUL.md"), "w").write("You are a capable autonomous coding agent. Work without asking the user questions: nobody is watching this session.\n")
print(home)
