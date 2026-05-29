"""Multi-step loop stability: the model plans -> code action -> observation -> repeat,
in a deterministic mock environment, until the goal or a step cap."""
import json, time
from pathlib import Path

from lib.evals.base import LLMClient, save_checkpoint, load_checkpoint
from lib.agentic.base import extract_code, load_system_prompt, build_tool_doc
from lib.agentic.sandbox import run_code_action

def detect_derailment(actions: list[str]) -> bool:
    if len(actions) >= 3 and actions[-1] == actions[-2] == actions[-3]:
        return True
    return False

class MultiStepEval:
    def __init__(self, client: LLMClient, system_prompt: str, data_dir: Path,
                 limit: int | None = None, results_dir: Path | None = None,
                 max_tokens: int = 768, max_steps: int = 8):
        self.client = client
        self.system_prompt = system_prompt
        self.data_dir = Path(data_dir)
        self.limit = limit
        self.results_dir = Path(results_dir) if results_dir else None
        self.max_tokens = max_tokens
        self.max_steps = max_steps

    def _items(self):
        items = [json.loads(l) for l in (self.data_dir / "multistep.jsonl").read_text().splitlines() if l.strip()]
        return items[:self.limit] if self.limit else items

    def _run_episode(self, item) -> dict:
        cap = item.get("max_steps", self.max_steps)
        history, actions = [], []
        for step in range(cap):
            user = (f"{build_tool_doc()}\n\nGoal: {item['task']}\n"
                    f"History:\n" + ("\n".join(history) or "(none)") +
                    "\n\nEmit the NEXT single ```python``` action; assign `result`. "
                    "When the goal is fully achieved, that final action is the completion.")
            resp = self.client.chat(
                [{"role": "system", "content": self.system_prompt},
                 {"role": "user", "content": user}], max_tokens=self.max_tokens)
            code = extract_code(resp)
            actions.append(code)
            out = run_code_action(code)
            obs = f"step {step}: ok={out.ok} result={str(out.result)[:120]} err={out.error[:80]}"
            history.append(obs)
            done = (item["goal_tool"] in code and item["goal_arg_contains"] in code and out.ok)
            if done:
                return {"completed": True, "steps": step + 1, "derailed": False}
            if detect_derailment(actions):
                return {"completed": False, "steps": step + 1, "derailed": True}
        return {"completed": False, "steps": cap, "derailed": True}

    def evaluate(self) -> dict:
        items = self._items()
        ckpt = self.results_dir / "multistep_progress.json" if self.results_dir else None
        state = load_checkpoint(ckpt) or {"episodes": {}}
        t0 = time.time()
        for item in items:
            if item["id"] in state["episodes"]:
                continue
            state["episodes"][item["id"]] = self._run_episode(item)
            save_checkpoint(ckpt, state)
        eps = list(state["episodes"].values())
        n = len(eps)
        completed = sum(1 for e in eps if e["completed"])
        score = completed / n * 100 if n else 0
        avg_steps = round(sum(e["steps"] for e in eps) / n, 2) if n else 0
        results = {"score": round(score, 2), "metric": "completion_rate",
                   "completed": completed, "total": n, "avg_steps": avg_steps,
                   "derailed": sum(1 for e in eps if e["derailed"]),
                   "elapsed_s": round(time.time() - t0, 1)}
        print(f"[multistep] {completed}/{n} completed, avg_steps={avg_steps}", flush=True)
        if self.results_dir:
            (self.results_dir / "multistep_detail.json").write_text(json.dumps(results, indent=2))
        return results

if __name__ == "__main__":
    import argparse
    from lib.config import get
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()
    sp = load_system_prompt(get("agentic.system_prompt"))
    with LLMClient(args.api_base, args.model, think=True) as c:
        ev = MultiStepEval(c, sp, get("agentic.data_dir"), limit=args.limit,
                           results_dir=Path(args.results_dir) if args.results_dir else None)
        print(ev.evaluate())
