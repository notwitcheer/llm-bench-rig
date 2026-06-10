"""Commit-the-fix axis. A per-run mutating tool `apply_fix(target, value)` writes to an
isolated world_state; everything else delegates to the normal dispatch. Verification is
state-based (check_commit), mirroring SWE-bench grading the resulting diff rather than the
model's prose: a model that only *describes* a fix never touches world_state and fails."""
from lib.agentic.native.dispatch import dispatch_tool


def make_commit_dispatch(base=dispatch_tool):
    """Return (dispatch, world_state). apply_fix(target, value) sets world_state[target]=str(value)
    and returns ok; any other tool delegates to `base`. Fresh state per call -> isolated per run."""
    world_state = {}

    def dispatch(name, args):
        if name == "apply_fix":
            world_state[str(args.get("target", ""))] = str(args.get("value", ""))
            return {"ok": True, "result": "ok", "error": ""}
        return base(name, args)

    return dispatch, world_state


def check_commit(task: dict, world_state: dict) -> bool:
    """Success iff the model committed the expected value via apply_fix (state-verified)."""
    return str(world_state.get(task["target"], "")).strip() == str(task["expect"]).strip()
