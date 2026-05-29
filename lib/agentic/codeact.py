"""CodeAct evaluator: the model writes Python that orchestrates mock tools to
accomplish a task. Scored pass@1 by executing the code in the sandbox."""
import json, time
from pathlib import Path

from lib.evals.base import LLMClient, save_checkpoint, load_checkpoint
from lib.agentic.base import extract_code, load_system_prompt, build_tool_doc
from lib.agentic.sandbox import run_code_action

def check_result(result, check: dict) -> bool:
    t, v = check["type"], check["value"]
    if t == "equals":
        return result == v or str(result) == str(v)
    if t == "startswith":
        return str(result).startswith(str(v))
    if t == "contains":
        return str(v) in str(result)
    if t == "approx":
        try:
            return abs(float(result) - float(v)) <= float(check.get("tol", 1e-6))
        except (TypeError, ValueError):
            return False
    raise ValueError(f"unknown check type: {t}")

def _build_messages(system_prompt: str, task: str) -> list[dict]:
    user = (f"{build_tool_doc()}\n\nTask: {task}\n\n"
            "Write a single ```python``` code block that completes the task and "
            "assigns the final answer to a variable named `result`.")
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": user}]

class CodeActEval:
    def __init__(self, client: LLMClient, system_prompt: str, data_dir: Path,
                 limit: int | None = None, results_dir: Path | None = None,
                 max_tokens: int = 1536, exec_timeout: int = 10):
        self.client = client
        self.system_prompt = system_prompt
        self.data_dir = Path(data_dir)
        self.limit = limit
        self.results_dir = Path(results_dir) if results_dir else None
        self.max_tokens = max_tokens
        self.exec_timeout = exec_timeout

    def _items(self) -> list[dict]:
        items = [json.loads(l) for l in (self.data_dir / "codeact.jsonl").read_text().splitlines() if l.strip()]
        return items[:self.limit] if self.limit else items

    def evaluate(self) -> dict:
        items = self._items()
        ckpt_path = self.results_dir / "codeact_progress.json" if self.results_dir else None
        state = load_checkpoint(ckpt_path) or {"done": {}, "first_tool": {}}
        passed = sum(1 for v in state["done"].values() if v)
        t0 = time.time(); n = len(items)

        for i, item in enumerate(items):
            if item["id"] in state["done"]:
                continue
            msgs = _build_messages(self.system_prompt, item["task"])
            resp = self.client.chat(msgs, max_tokens=self.max_tokens)
            code = extract_code(resp)
            out = run_code_action(code, timeout=self.exec_timeout)
            ok = bool(out.ok and check_result(out.result, item["check"]))
            state["done"][item["id"]] = ok
            if ok:
                passed += 1
            save_checkpoint(ckpt_path, state)
            done = len([1 for _ in state["done"]])
            if done % 10 == 0 or done == n:
                rate = done / (time.time() - t0) if time.time() > t0 else 0
                print(f"[codeact] {passed}/{done} pass [{done}/{n}] {rate:.2f} q/s", flush=True)

        score = passed / n * 100 if n else 0
        results = {"score": round(score, 2), "metric": "pass@1",
                   "passed": passed, "total": n,
                   "by_tier": _tier_breakdown(items, state["done"]),
                   "elapsed_s": round(time.time() - t0, 1)}
        if self.results_dir:
            (self.results_dir / "codeact_detail.json").write_text(json.dumps(results, indent=2))
        return results

def _tier_breakdown(items, done) -> dict:
    out = {}
    for it in items:
        tier = it.get("tier", "?")
        d = out.setdefault(tier, {"passed": 0, "total": 0})
        d["total"] += 1
        if done.get(it["id"]):
            d["passed"] += 1
    return out

if __name__ == "__main__":
    import argparse
    from lib.config import get
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", default="http://127.0.0.1:8090/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--results-dir", default=None)
    args = ap.parse_args()
    from lib.agentic.base import CODEACT_SYSTEM_PROMPT
    with LLMClient(args.api_base, args.model, think=True) as c:
        ev = CodeActEval(c, CODEACT_SYSTEM_PROMPT, get("agentic.data_dir"), limit=args.limit,
                         results_dir=Path(args.results_dir) if args.results_dir else None)
        print(ev.evaluate())
