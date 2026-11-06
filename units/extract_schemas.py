#!/usr/bin/env python3
"""extract the REAL Hermes tool schemas used by the units experiment, verbatim, from a pinned Hermes checkout.
usage (on the machine holding the checkout, with the checkout's own python):
    cd <checkout> && venv/bin/python /path/to/extract_schemas.py <out.json>
writes {pin, tools: {name: {schema, file, schema_line, arg_line, arg, unit, note}}}. Nothing is edited: the schema dicts are the objects the
checkout itself registers (or, for discord, the checkout's own _build_schema over every action it lists)."""
import json
import os
import re
import subprocess
import sys

OUT = sys.argv[1]
ROOT = os.getcwd()
sys.path.insert(0, ROOT)
pin = subprocess.run(["git", "rev-parse", "--short=7", "HEAD"], capture_output=True, text=True).stdout.strip()


def find_line(path, needle, start=1):
    for i, line in enumerate(open(os.path.join(ROOT, path), encoding="utf-8"), 1):
        if i >= start and needle in line:
            return i
    return None


tools = {}
import tools.terminal_tool as t  # noqa: E402
import tools.process_registry as p  # noqa: E402
import tools.browser_cdp_tool as c  # noqa: E402
import tools.browser_use_cli as b  # noqa: E402
import tools.computer_use.schema as cu  # noqa: E402
import tools.discord_tool as d  # noqa: E402

tools["terminal"] = dict(schema=t.TERMINAL_SCHEMA, file="tools/terminal_tool.py", schema_line=find_line("tools/terminal_tool.py", "TERMINAL_SCHEMA = {"),
                         arg="timeout", arg_line=find_line("tools/terminal_tool.py", '"timeout": {', find_line("tools/terminal_tool.py", "TERMINAL_SCHEMA = {")), unit="seconds",
                         note="description text is the checkout's own (f-string with FOREGROUND_MAX_TIMEOUT resolved at import)")
tools["process"] = dict(schema=p.PROCESS_SCHEMA, file="tools/process_registry.py", schema_line=find_line("tools/process_registry.py", "PROCESS_SCHEMA = {"),
                        arg="timeout", arg_line=find_line("tools/process_registry.py", '"timeout": {', find_line("tools/process_registry.py", "PROCESS_SCHEMA = {")), unit="seconds",
                        note="applies to action 'wait'")
tools["browser_cdp"] = dict(schema=c.BROWSER_CDP_SCHEMA, file="tools/browser_cdp_tool.py", schema_line=find_line("tools/browser_cdp_tool.py", "BROWSER_CDP_SCHEMA: dict"),
                            arg="timeout", arg_line=find_line("tools/browser_cdp_tool.py", '"timeout": {"type": "number"'), unit="seconds", note="")
tools["browser_exec"] = dict(schema=b.BROWSER_EXEC_SCHEMA, file="tools/browser_use_cli.py", schema_line=find_line("tools/browser_use_cli.py", "BROWSER_EXEC_SCHEMA = {"),
                             arg="timeout_s", arg_line=find_line("tools/browser_use_cli.py", '"timeout_s": {'), unit="seconds",
                             note="STATIC FALLBACK description (the managed CLI is not installed on this checkout); the unit-naming argument description is the live one")
tools["computer_use"] = dict(schema=cu.COMPUTER_USE_SCHEMA, file="tools/computer_use/schema.py", schema_line=find_line("tools/computer_use/schema.py", "COMPUTER_USE_SCHEMA: dict"),
                             arg="seconds", arg_line=find_line("tools/computer_use/schema.py", '"seconds": {'), unit="seconds", note="applies to action 'wait'")
actions = [name for name, _fn, _sig, _desc in d._ACTION_MANIFEST]
tools["discord"] = dict(schema=d._build_schema(actions, {}, tool_name="discord"), file="tools/discord_tool.py", schema_line=find_line("tools/discord_tool.py", "def _build_schema("),
                        arg="auto_archive_duration", arg_line=find_line("tools/discord_tool.py", '"auto_archive_duration": {'), unit="minutes",
                        note="built by the checkout's own _build_schema over every action in _ACTION_MANIFEST with empty capabilities; applies to create_thread")
json.dump({"pin": pin, "tools": tools}, open(OUT, "w"), indent=1, sort_keys=False)
for k, v in tools.items():
    arg = v["schema"]["parameters"]["properties"][v["arg"]]
    print(k, v["file"], v["schema_line"], v["arg_line"], json.dumps(arg)[:150])
