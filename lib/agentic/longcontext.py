"""Long-context use under agent load: bury a needle in a long simulated
memory/tool-output context, then require the model to use it."""
import json, time
from pathlib import Path

from lib.evals.base import LLMClient, save_checkpoint, load_checkpoint
from lib.agentic.base import extract_code, load_system_prompt, build_tool_doc
from lib.agentic.codeact import check_result
from lib.agentic.sandbox import run_code_action

_FILLER = "[mem] tool_log: routine status ok; cache warm; no action needed. "

def approx_tokens(text: str) -> int:
    return len(text) // 4

def build_haystack(needle: str, target_tokens: int, depth: float = 0.5) -> str:
    target_chars = target_tokens * 4
    pre_chars = int(target_chars * depth)
    post_chars = target_chars - pre_chars
    def fill(nchars):
        buf = []
        c = 0
        while c < nchars:
            buf.append(_FILLER); c += len(_FILLER)
        return "".join(buf)
    return fill(pre_chars) + f"\n[mem] IMPORTANT: {needle}\n" + fill(post_chars)

class LongContextUseEval:
    def __init__(self, client: LLMClient, system_prompt: str, data_dir: Path,
                 depths: list[int], limit: int | None = None,
                 results_dir: Path | None = None, max_tokens: int = 512,
                 exec_timeout: int = 10):
        self.client = client
        self.system_prompt = system_prompt
        self.data_dir = Path(data_dir)
        self.depths = depths
        self.limit = limit
        self.results_dir = Path(results_dir) if results_dir else None
        self.max_tokens = max_tokens
        self.exec_timeout = exec_timeout

    def _items(self):
        items = [json.loads(l) for l in (self.data_dir / "longcontext.jsonl").read_text().splitlines() if l.strip()]
        return items[:self.limit] if self.limit else items

    def evaluate(self) -> dict:
        items = self._items()
        ckpt = self.results_dir / "longcontext_progress.json" if self.results_dir else None
        state = load_checkpoint(ckpt) or {"done": {}}  # key "depth:id" -> bool
        t0 = time.time()
        per_depth = {d: {"passed": 0, "total": 0} for d in self.depths}
        for depth in self.depths:
            for item in items:
                key = f"{depth}:{item['id']}"
                per_depth[depth]["total"] += 1
                if key in state["done"]:
                    if state["done"][key]:
                        per_depth[depth]["passed"] += 1
                    continue
                hay = build_haystack(item["needle"], target_tokens=depth)
                user = (f"{build_tool_doc()}\n\nContext (memory + tool logs):\n{hay}\n\n"
                        f"Task: {item['question']}\nWrite a ```python``` block; assign `result`.")
                try:
                    resp = self.client.chat(
                        [{"role": "system", "content": self.system_prompt},
                         {"role": "user", "content": user}], max_tokens=self.max_tokens)
                    out = run_code_action(extract_code(resp), timeout=self.exec_timeout)
                    ok = bool(out.ok and check_result(out.result, item["check"]))
                except Exception:
                    ok = False  # OOM / context overflow -> scored 0 at this depth
                state["done"][key] = ok
                if ok:
                    per_depth[depth]["passed"] += 1
                save_checkpoint(ckpt, state)
            d = per_depth[depth]
            print(f"[longcontext] {depth//1024}K: {d['passed']}/{d['total']}", flush=True)

        scores = {f"{d//1024}K": round(100 * v["passed"] / v["total"], 2) if v["total"] else 0
                  for d, v in per_depth.items()}
        overall = round(sum(scores.values()) / len(scores), 2) if scores else 0
        results = {"score": overall, "metric": "retrieve_and_use",
                   "by_depth": scores, "elapsed_s": round(time.time() - t0, 1)}
        if self.results_dir:
            (self.results_dir / "longcontext_detail.json").write_text(json.dumps(results, indent=2))
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
    depths = get("agentic.evals.longcontext.depths", [16384, 32768, 65536])
    with LLMClient(args.api_base, args.model, think=False, timeout=300) as c:
        ev = LongContextUseEval(c, sp, get("agentic.data_dir"), depths=depths,
                                limit=args.limit,
                                results_dir=Path(args.results_dir) if args.results_dir else None)
        print(ev.evaluate())
