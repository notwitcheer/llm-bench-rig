"""SWE-bench reality anchor. solve_instance() runs our native tool-calling loop with real
repo tools against a checked-out repo (any RepoBackend) and extracts the git diff as the
candidate patch. The capsule orchestration (main) wraps this with per-instance Docker
containers + writes predictions; the official swebench harness grades the patches."""
import json
import subprocess
import sys
from pathlib import Path

from lib.agentic.native.tools_repo import make_repo_tools, make_dispatch
from lib.agentic.native.agent_loop import run_agent

GOAL = ("You are fixing a bug in a Python repository checked out at the repo root. The issue:\n\n"
        "{problem}\n\n"
        "Use list_dir, search, and read_file to locate the cause, fix it with edit_file, and verify by "
        "running the project's tests with run_bash. Do NOT edit test files. When the fix is complete, "
        "reply with a one-line summary and stop.")


def solve_instance(client, problem_statement: str, backend, max_steps: int = 40) -> dict:
    schemas, ns = make_repo_tools(backend)
    err = ""
    try:
        res = run_agent(client, GOAL.format(problem=problem_statement), schemas,
                        max_steps=max_steps, dispatch=make_dispatch(ns))
        tel = {"final_text": res.final_text, "n_steps": res.n_steps,
               "n_tool_calls": res.n_tool_calls, "n_tokens": res.n_tokens, "stalled": res.stalled}
    except Exception as e:
        # a serving error mid-loop (e.g. llama-server 500 on malformed tool-call JSON) must not
        # void the whole model — record it, keep any partial diff already made in the container.
        err = f"{type(e).__name__}: {e}"
        tel = {"final_text": "", "n_steps": -1, "n_tool_calls": -1, "n_tokens": 0, "stalled": True}
    patch, _ = backend.run("git add -A >/dev/null 2>&1; git diff --cached")
    return {**tel, "patch": patch, "patch_empty": not patch.strip(), "error": err}


def _docker(args, **kw):
    return subprocess.run(["docker"] + args, capture_output=True, text=True, **kw)


def run_one(slug: str, instance: dict, client, max_steps: int = 40) -> dict:
    """Start the instance's container, solve, extract patch, tear down. `instance` is a
    SWE-bench row (needs instance_id + problem_statement); image key resolved via swebench."""
    from lib.agentic.native.tools_repo import DockerRepoBackend
    from swebench.harness.test_spec.test_spec import make_test_spec
    iid = instance["instance_id"]
    # namespace="swebench" -> the prebuilt Docker Hub image (no local build needed)
    image = make_test_spec(instance, namespace="swebench").instance_image_key
    cid = _docker(["run", "-d", image, "sleep", "infinity"]).stdout.strip()
    try:
        backend = DockerRepoBackend(cid, root="/testbed")
        out = solve_instance(client, instance["problem_statement"], backend, max_steps=max_steps)
    finally:
        _docker(["rm", "-f", cid])
    return {"instance_id": iid, "model_name_or_path": slug, "model_patch": out["patch"],
            "_telemetry": {k: out[k] for k in ("n_steps", "n_tool_calls", "n_tokens", "stalled", "patch_empty")}}


def main(slug: str, instance_ids_file: str):
    """Generate predictions for one served model over the pinned instance_ids."""
    from lib.agentic.native.client import LlamaClient
    from datasets import load_dataset
    ids = [x.strip() for x in Path(instance_ids_file).read_text().split() if x.strip()]
    rows = {r["instance_id"]: r for r in load_dataset("princeton-nlp/SWE-bench_Verified", split="test")}
    client = LlamaClient(timeout=600)
    pred_dir = Path("predictions"); pred_dir.mkdir(exist_ok=True)
    tel_dir = Path(f"results/swebench/{slug}"); tel_dir.mkdir(parents=True, exist_ok=True)
    preds, tel = [], []
    for iid in ids:
        rec = run_one(slug, rows[iid], client)
        tel.append({"instance_id": iid, **rec.pop("_telemetry")})
        preds.append(rec)
        print(f"[swebench:{slug}] {iid:40} patch_empty={tel[-1]['patch_empty']} steps={tel[-1]['n_steps']}")
    (pred_dir / f"{slug}.jsonl").write_text("\n".join(json.dumps(p) for p in preds) + "\n")
    (tel_dir / "telemetry.json").write_text(json.dumps(tel, indent=2))
    print(f"wrote predictions/{slug}.jsonl ({len(preds)})")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
