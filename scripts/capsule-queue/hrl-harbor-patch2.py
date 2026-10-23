"""Second step for an already-patched Harbor hermes adapter: v-prefix numeric versions for the git ref
and fetch install.sh from that ref. Unlinks before writing so a uv-cache hardlink is not edited."""
import harbor.agents.installed.hermes as m
from pathlib import Path

p = Path(m.__file__)
s = p.read_text()
if "ref = self._version" in s:
    print("ref patch already present")
    raise SystemExit
old_b = '        branch_flag = f" --branch {self._version}" if self._version else ""\n'
new_b = (
    "        # LOCAL PATCH hrl: OpenEnv pins numeric versions (2026.9.24); Hermes tags are v-prefixed.\n"
    "        ref = self._version\n"
    "        if ref and ref[0].isdigit():\n"
    '            ref = "v" + ref\n'
    '        branch_flag = f" --branch {ref}" if ref else ""\n'
)
assert old_b in s
s = s.replace(old_b, new_b, 1)
s = s.replace("{self._version or 'main'}/scripts/install.sh", "{ref or 'main'}/scripts/install.sh", 1)
assert "{ref or 'main'}" in s
p.unlink()
p.write_text(s)
print("ok")
