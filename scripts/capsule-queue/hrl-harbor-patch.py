"""Local fix for Harbor's hermes adapter (harbor 0.24.0), capsule hrl 2026-10-05.

Three bugs reproduced in clean python:3.12-slim containers:
  1. main's installer: bundled Node fails its check with exit 127 (libatomic.so.1 missing);
     the tag installer also needs a C++ toolchain. -> add libatomic + build_tools deps.
  2. install.sh is always fetched from main even when a version is pinned. -> fetch from the ref.
  3. `hermes version` is not a Hermes command (docs: `hermes --version`). -> fix both call sites.
Idempotent.
"""
import harbor.agents.installed.hermes as m
from pathlib import Path

p = Path(m.__file__)
s = p.read_text()
if "LOCAL PATCH hrl" in s:
    print("already patched", p)
    raise SystemExit
s = s.replace(
    "from harbor.agents.installed.base import (\n    BaseInstalledAgent,\n",
    "from harbor.agents.installed.base import (\n    BaseInstalledAgent,\n    PackageSpec,\n",
    1,
)
s = s.replace(
    "    options_model = HermesOptions\n",
    "    options_model = HermesOptions\n\n"
    "    # LOCAL PATCH hrl: Node in main's installer needs libatomic1 (exit 127 without it).\n"
    "    SYSTEM_PACKAGES = {\n"
    "        **BaseInstalledAgent.SYSTEM_PACKAGES,\n"
    "        \"libatomic\": PackageSpec(\n"
    "            commands=(),\n"
    "            packages={\"apt-get\": (\"libatomic1\",), \"dnf\": (\"libatomic\",), \"yum\": (\"libatomic\",), \"apk\": (\"libatomic\",)},\n"
    "            always_install=True,\n"
    "        ),\n"
    "    }\n",
    1,
)
s = s.replace(
    '            environment, ("curl", "git", "ripgrep", "xz")\n',
    '            environment, ("curl", "git", "ripgrep", "xz", "build_tools", "libatomic")\n',
    1,
)
s = s.replace(
    '        branch_flag = f" --branch {self._version}" if self._version else ""\n',
    "        # LOCAL PATCH hrl: OpenEnv pins numeric versions (2026.9.24); Hermes tags are v-prefixed.\n"
    "        ref = self._version\n"
    "        if ref and ref[0].isdigit():\n"
    "            ref = \"v\" + ref\n"
    '        branch_flag = f" --branch {ref}" if ref else ""\n',
    1,
)
s = s.replace(
    "https://raw.githubusercontent.com/NousResearch/hermes-agent/main/scripts/install.sh",
    "https://raw.githubusercontent.com/NousResearch/hermes-agent/{ref or 'main'}/scripts/install.sh",
    1,
)
s = s.replace("hermes version'", "hermes --version'", 1)
s = s.replace('                "hermes version"\n', '                "hermes --version"\n', 1)
for needle in ("PackageSpec,", '"libatomic"', "build_tools", "{ref or 'main'}", "ref = self._version", "hermes --version'", '"hermes --version"'):
    assert needle in s, needle
assert "hermes version" not in s
p.write_text(s)
print("patched", p)
