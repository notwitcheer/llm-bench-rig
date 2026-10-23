"""Local patches for the hermes-harness transfer check (capsule, 2026-10-05).

1. OpenEnv seams.py on main has no `hermes` seam (comment: removed after 5/5 install failures,
   exit 127). Re-add one locally so `harness="hermes"` routes through the capture proxy.
2. DirectExposure advertises 127.0.0.1, which a docker sandbox cannot reach. Let
   OPENENV_DIRECT_HOST override it (docker0 gateway 172.17.0.1 on capsule).
3. A copy of FineEnvs eval/evaluate.py whose harness list and sandbox come from env vars
   (HRL_HARNESSES, HRL_SANDBOX); everything else (temperature 0.8, 4096 tokens, 17 steps,
   600 s agent timeout, grader) is byte-identical.
Idempotent: re-running is a no-op.
"""

import re
from pathlib import Path

ROOT = Path.home() / "harness-rl/FineEnvs/05-multi-harness-rl"
OE = ROOT / ".deps/OpenEnv/src/openenv"

seams = OE / "harbor/seams.py"
s = seams.read_text()
if '"hermes": Seam(' not in s:
    block = '''    "hermes": Seam(
        name="hermes",
        status="untested",
        dialect="openai_chat",
        model_fmt="openai/{model}",
        # LOCAL PATCH (capsule hrl). Harbor's hermes.run() picks `--provider openai-api` only when
        # OPENAI_API_KEY is in the MANAGER process env, and forwards OPENAI_BASE_URL from there too.
        # Both are constant across sessions (one proxy), so the process channel is race-free; the
        # per-session key rides agent_env. SmolDataEnvs' grader is numeric-only, no LLM judge.
        env={"OPENAI_API_KEY": "openenv-capture", "OPENAI_BASE_URL": "{base_url}/v1"},
        agent_env={"OPENAI_BASE_URL": "{base_url}/v1", "OPENAI_API_KEY": "{session}"},
        step_limit=lambda n: {"max_turns": n},
    ),
'''
    s = s.replace('    "vibe": Seam(\n', block + '    "vibe": Seam(\n', 1)
    assert '"hermes": Seam(' in s
    seams.write_text(s)
    print("patched seams.py")

fwd = OE / "core/harness/capture/forwarding.py"
f = fwd.read_text()
old = '    def __init__(self, advertise_host: str = "127.0.0.1", scheme: str = "http") -> None:\n        super().__init__()\n        self._host = advertise_host\n'
new = '    def __init__(self, advertise_host: str = "127.0.0.1", scheme: str = "http") -> None:\n        super().__init__()\n        import os  # LOCAL PATCH (capsule hrl): docker sandboxes cannot reach 127.0.0.1\n        self._host = os.environ.get("OPENENV_DIRECT_HOST", advertise_host)\n'
if "OPENENV_DIRECT_HOST" not in f:
    assert old in f, "DirectExposure signature moved"
    fwd.write_text(f.replace(old, new, 1))
    print("patched forwarding.py")

ev = (ROOT / "eval/evaluate.py").read_text()
out = ROOT / "eval/evaluate_hrl.py"
e = ev.replace(
    'HARNESSES = ("opencode", "claude-code", "codex", "mini-swe-agent")',
    'import os\nHARNESSES = tuple(os.environ.get("HRL_HARNESSES", "hermes").split(","))',
    1,
).replace('sandbox="daytona",', 'sandbox=os.environ.get("HRL_SANDBOX", "docker"),', 1)
assert "HRL_HARNESSES" in e and "HRL_SANDBOX" in e
out.write_text(e)
print("wrote", out)
