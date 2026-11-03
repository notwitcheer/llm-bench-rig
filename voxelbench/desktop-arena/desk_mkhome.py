#!/usr/bin/env python3
"""desk_mkhome.py <root> <agent> <port> <model-alias> <ctx> [image]

ONE entrant's private HERMES_HOME under <root>/home-<agent>/ (a directory of its own, NOT a profile of the main install: the main hermes-gateway.service
multiplexes every `hermes -p <profile>` and must never see these agents). Launch with
  HERMES_DISABLE_LAZY_INSTALLS=1 HERMES_HOME=<root>/home-<agent> <checkout>/.hermes/bin/hermes ...
installs/ and tools/ are symlinked from the main root (shared read-only runtime, see four_mkhome.py in voxelbench/four-agents).
terminal.backend docker + the desktop sandbox image: computer_use, the screen (Xvnc + Xfce) and cua-driver all run INSIDE the container
(docs: user-guide/features/bot-screen 'Where the screen runs'). Memory off, telemetry off, approvals STANDARD with the background grants for every computer_use action pre-listed in command_allowlist (keys cua:<action>:<background|foreground>; the pin's cua-driver answers key presses on GTK apps with 'Background delivery is not available ... retry with delivery_mode:foreground', so the foreground grants are listed too, docs 'Safety'). every entrant gets the same pre-granted actions; toolset computer_use ONLY (pure desktop agent: no terminal/file tool to shortcut a task) and tools.tool_search.enabled off (default config defers computer_use behind tool_search/tool_describe/tool_call, measured on the pin 2026-11-03: the model sees only those 3 bridge tools).
"""
import os
import sys

root, ag, port, alias, ctx = sys.argv[1:6]
image = sys.argv[6] if len(sys.argv) > 6 else "nousresearch/hermes-sandbox:desktop"
home = os.path.join(root, "home-" + ag)
os.makedirs(home, exist_ok=True)
os.chmod(home, 0o700)
ACTIONS = ["click", "double_click", "right_click", "middle_click", "drag", "scroll", "type", "key", "set_value", "wait", "focus_app", "bring_to_front"]
ALLOW = "[" + ", ".join("\"cua:%s:%s\"" % (a, m) for a in ACTIONS for m in ("background", "foreground")) + "]"
cfg = """model:
  default: %(alias)s
  provider: custom
  base_url: http://127.0.0.1:%(port)s/v1
  api_key: local-key
  context_length: %(ctx)s
toolsets:
  - computer_use
tools:
  tool_search:
    enabled: "off"
terminal:
  backend: docker
  docker_image: %(image)s
  container_persistent: true
  docker_orphan_reaper: false
agent:
  max_turns: 60
  clarify_timeout: 20
approvals:
  mode: manual
command_allowlist: %(allow)s
memory:
  memory_enabled: false
  user_profile_enabled: false
display:
  interface: cli
  compact: true
  streaming: true
telemetry:
  shared_metrics:
    enabled: false
    send: false
    offer_version: 2
""" % {"alias": alias, "port": port, "ctx": ctx, "image": image, "allow": ALLOW}
open(os.path.join(home, "config.yaml"), "w").write(cfg)
main = os.environ.get("FR_MAIN_HOME", os.path.expanduser("~/.hermes"))
for sub in ("installs", "tools"):
    dst = os.path.join(home, sub)
    if not os.path.lexists(dst):
        os.symlink(os.path.join(main, sub), dst)
open(os.path.join(home, "SOUL.md"), "w").write("You operate a Linux desktop through the computer_use tool. Work without asking the user questions: nobody is watching this session.\n")
print(home)
