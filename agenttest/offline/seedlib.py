"""seedlib: seed the 10 frozen v1 task folders into ANY host directory with the {WORK} placeholder rendered as the CONTAINER path.

why: gen_seeds.build() refuses to wipe anything outside /opt/data/tmp/ (a mercury-only rule) and bakes the work path it is given into t09/run_report.sh. The offline
harness seeds on the capsule host but the agent and the checkers see the folder at /opt/data/tmp/sparkagent/work inside the container, so the placeholder must be
rendered with that path. Same files, same modes (0644), same dirs as gen_seeds.build(); `equivalence` in offline_run.py proves it with a container run of the real build().
"""
import hashlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V1 = os.path.dirname(HERE)
sys.path.insert(0, V1)
import gen_seeds  # noqa: E402

CONTAINER_WORK = "/opt/data/tmp/sparkagent/work"


def seed(dest, container_work=CONTAINER_WORK):
    """write the seed tree into dest (must not hold task files yet). returns {rel: sha256} like gen_seeds.build()."""
    os.makedirs(os.path.join(dest, "smoke"), exist_ok=True)
    for t in gen_seeds.TASK_IDS:
        os.makedirs(os.path.join(dest, t), exist_ok=True)
    manifest = {}
    for rel, raw in gen_seeds.seed_files():
        p = os.path.join(dest, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        b = gen_seeds.render(raw, container_work)
        with open(p, "wb") as f:
            f.write(b)
        os.chmod(p, 0o644)
        manifest[rel] = hashlib.sha256(b).hexdigest()
    return manifest


if __name__ == "__main__":
    import json
    print(json.dumps(seed(sys.argv[1]), indent=1))
