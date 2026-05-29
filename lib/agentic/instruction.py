"""Instruction-following under load: each instruction is run twice — once with the
full Hermes system prompt, once with a minimal one. Reports compliance under load
and the delta (capability tax of the heavy prompt)."""
import json, re, time
from pathlib import Path

from lib.evals.base import LLMClient, save_checkpoint, load_checkpoint
from lib.agentic.base import load_system_prompt

_MINIMAL = "You are a helpful assistant. Follow the user's instruction exactly."

def check_compliance(text: str, check: dict) -> bool:
    t = (text or "").strip()
    tag = "</think>"
    if tag in t:
        t = t[t.rfind(tag) + len(tag):].strip()
    typ, v = check["type"], check["value"]
    if typ == "exact":
        return t == v
    if typ == "regex":
        return re.fullmatch(v, t) is not None or re.search(v, t) is not None
    if typ == "json_keys":
        try:
            obj = json.loads(t)
            return all(k in obj for k in v)
        except json.JSONDecodeError:
            return False
    if typ == "forbids":
        return v.lower() not in t.lower()
    raise ValueError(f"unknown check type: {typ}")

class InstructionLoadEval:
    def __init__(self, client: LLMClient, system_prompt: str, data_dir: Path,
                 limit: int | None = None, results_dir: Path | None = None,
                 max_tokens: int = 512):
        self.client = client
        self.heavy = system_prompt
        self.data_dir = Path(data_dir)
        self.limit = limit
        self.results_dir = Path(results_dir) if results_dir else None
        self.max_tokens = max_tokens

    def _items(self):
        items = [json.loads(l) for l in (self.data_dir / "instruction.jsonl").read_text().splitlines() if l.strip()]
        return items[:self.limit] if self.limit else items

    def _ask(self, system: str, instruction: str) -> str:
        return self.client.chat(
            [{"role": "system", "content": system},
             {"role": "user", "content": instruction}],
            max_tokens=self.max_tokens)

    def evaluate(self) -> dict:
        items = self._items()
        ckpt = self.results_dir / "instruction_progress.json" if self.results_dir else None
        state = load_checkpoint(ckpt) or {"loaded": {}, "minimal": {}}
        t0 = time.time(); n = len(items)
        for item in items:
            iid = item["id"]
            if iid not in state["loaded"]:
                state["loaded"][iid] = check_compliance(self._ask(self.heavy, item["instruction"]), item["check"])
                save_checkpoint(ckpt, state)
            if iid not in state["minimal"]:
                state["minimal"][iid] = check_compliance(self._ask(_MINIMAL, item["instruction"]), item["check"])
                save_checkpoint(ckpt, state)
        loaded = sum(1 for v in state["loaded"].values() if v)
        minimal = sum(1 for v in state["minimal"].values() if v)
        load_score = loaded / n * 100 if n else 0
        min_score = minimal / n * 100 if n else 0
        results = {"score": round(load_score, 2), "metric": "compliance@load",
                   "compliance_loaded": round(load_score, 2),
                   "compliance_minimal": round(min_score, 2),
                   "delta": round(min_score - load_score, 2),
                   "total": n, "elapsed_s": round(time.time() - t0, 1)}
        print(f"[instruction] loaded={load_score:.1f}% minimal={min_score:.1f}% "
              f"delta={results['delta']:.1f}", flush=True)
        if self.results_dir:
            (self.results_dir / "instruction_detail.json").write_text(json.dumps(results, indent=2))
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
    with LLMClient(args.api_base, args.model, think=False) as c:
        ev = InstructionLoadEval(c, sp, get("agentic.data_dir"), limit=args.limit,
                                 results_dir=Path(args.results_dir) if args.results_dir else None)
        print(ev.evaluate())
