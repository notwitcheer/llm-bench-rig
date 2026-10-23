"""Third step: skip the per-trial install when Hermes is already on PATH (baked image). Unlink-before-write."""
import harbor.agents.installed.hermes as m
from pathlib import Path
p = Path(m.__file__)
s = p.read_text()
if "LOCAL PATCH hrl-skip" in s:
    print("skip patch already present"); raise SystemExit
old = '                f"curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/{ref or \'main\'}/scripts/install.sh | bash -s -- --skip-setup{branch_flag} && "\n'
new = ('                # LOCAL PATCH hrl-skip: baked image already has the pinned Hermes.\n'
       '                \'export PATH="$HOME/.local/bin:$PATH"; if ! hermes --version >/dev/null 2>&1; then \'\n'
       '                f"curl -fsSL https://raw.githubusercontent.com/NousResearch/hermes-agent/{ref or \'main\'}/scripts/install.sh | bash -s -- --skip-setup{branch_flag}; fi && "\n')
assert old in s, "install line not found"
s = s.replace(old, new, 1)
p.unlink(); p.write_text(s); print("ok")
