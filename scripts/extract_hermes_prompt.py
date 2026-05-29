"""Extract Hermes Agent's stable system-prompt constants from a cloned hermes-agent
repo and assemble them into data/agentic/hermes_system_prompt.txt.

ast-based (no import) so it needs none of Hermes's dependencies. Pulls the static
"stable tier" guidance blocks (identity + execute_code/tool-use enforcement + memory/
skills/etc.) in the order documented in website/docs/developer-guide/prompt-assembly.md.
The volatile tier (live memory snapshot, timestamps) is intentionally excluded — for
the benchmark we want the static stressor the model always carries.

Usage: python3 scripts/extract_hermes_prompt.py [path-to-hermes-src]   (run from rig root)
"""
import ast
import sys
from pathlib import Path

HERMES = Path(sys.argv[1]) if len(sys.argv) > 1 else (Path.home() / "hermes-src")
pb = (HERMES / "agent" / "prompt_builder.py").read_text()
tree = ast.parse(pb)

# stable-tier order (per the prompt-assembly dev doc)
ORDER = [
    "DEFAULT_AGENT_IDENTITY",
    "TOOL_USE_ENFORCEMENT_GUIDANCE",
    "OPENAI_MODEL_EXECUTION_GUIDANCE",
    "GOOGLE_MODEL_OPERATIONAL_GUIDANCE",
    "MEMORY_GUIDANCE",
    "SKILLS_GUIDANCE",
    "SESSION_SEARCH_GUIDANCE",
    "TASK_COMPLETION_GUIDANCE",
    "KANBAN_GUIDANCE",
    "HERMES_AGENT_HELP_GUIDANCE",
]

consts = {}
for node in tree.body:
    if isinstance(node, ast.Assign):
        for tg in node.targets:
            if isinstance(tg, ast.Name):
                try:
                    v = ast.literal_eval(node.value)
                except Exception:
                    continue
                if isinstance(v, str) and v.strip():
                    consts[tg.id] = v

parts, used = [], []
for name in ORDER:
    if name in consts:
        parts.append(consts[name])
        used.append(name)
# include any other *_GUIDANCE string constants for completeness
for name, v in consts.items():
    if name.endswith("_GUIDANCE") and name not in used:
        parts.append(v)
        used.append(name)

prompt = "\n\n".join(parts)
out = Path("data/agentic/hermes_system_prompt.txt")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(prompt)
print("used constants:", used)
print(f"chars={len(prompt)}  approx_tokens={len(prompt)//4}")
