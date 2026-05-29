"""Run the full agentic suite over the configured model field. tmux-friendly."""
import json, time
from pathlib import Path

from lib.config import get
from lib.quality import start_llama_server, stop_llama_server
from lib.evals.base import LLMClient
from lib.agentic.base import load_system_prompt, CODEACT_SYSTEM_PROMPT
from lib.agentic.codeact import CodeActEval
from lib.agentic.instruction import InstructionLoadEval
from lib.agentic.longcontext import LongContextUseEval
from lib.agentic.multistep import MultiStepEval
from lib.agentic.score import build_summary

def run_model(model: dict) -> dict:
    slug, path = model["slug"], str(Path(model["path"]).expanduser())
    cfg = get("agentic.evals")
    sp = load_system_prompt(get("agentic.system_prompt"))
    data_dir = Path(get("agentic.data_dir"))
    results_dir = Path(get("results_dir", "./results")) / slug
    results_dir.mkdir(parents=True, exist_ok=True)
    ctx = get("agentic.ctx_size", 65536)
    port = get("llama_cpp.server_port", 8090)
    api_base = f"http://127.0.0.1:{port}/v1"

    print(f"\n===== {slug} =====", flush=True)
    proc = start_llama_server(path, ctx_size=ctx)
    evals = {}
    try:
        with LLMClient(api_base, Path(path).stem, think=cfg["codeact"]["think"], timeout=300) as c:
            # Option A: codeact uses a light prompt (self-contained block); the real
            # Hermes prompt conditions models into incremental tool-call markup.
            evals["codeact"] = CodeActEval(c, CODEACT_SYSTEM_PROMPT, data_dir, limit=cfg["codeact"]["limit"],
                results_dir=results_dir, max_tokens=cfg["codeact"]["max_tokens"],
                exec_timeout=cfg["codeact"]["exec_timeout"]).evaluate()
            evals["multistep"] = MultiStepEval(c, sp, data_dir, limit=cfg["multistep"]["limit"],
                results_dir=results_dir, max_tokens=cfg["multistep"]["max_tokens"],
                max_steps=cfg["multistep"]["max_steps"]).evaluate()
        with LLMClient(api_base, Path(path).stem, think=cfg["instruction"]["think"], timeout=300) as c:
            evals["instruction"] = InstructionLoadEval(c, sp, data_dir,
                limit=cfg["instruction"]["limit"], results_dir=results_dir,
                max_tokens=cfg["instruction"]["max_tokens"]).evaluate()
            evals["longcontext"] = LongContextUseEval(c, sp, data_dir,
                depths=cfg["longcontext"]["depths"], limit=cfg["longcontext"]["limit"],
                results_dir=results_dir, max_tokens=cfg["longcontext"]["max_tokens"]).evaluate()
    finally:
        stop_llama_server(proc)

    summary = build_summary(slug, evals, get("agentic.weights"))
    (results_dir / "agentic.json").write_text(json.dumps(summary, indent=2))
    print(f"[{slug}] Hermes Pairing Score = {summary['pairing_score']}", flush=True)
    return summary

def main():
    models = get("agentic.models")
    t0 = time.time()
    summaries = [run_model(m) for m in models]
    summaries.sort(key=lambda s: s["pairing_score"], reverse=True)
    out = Path(get("results_dir", "./results")) / "hermes_pairing_leaderboard.json"
    out.write_text(json.dumps(summaries, indent=2))
    print(f"\n===== LEADERBOARD ({(time.time()-t0)/60:.0f} min) =====")
    for rank, s in enumerate(summaries, 1):
        print(f"{rank}. {s['model']:28s} {s['pairing_score']:.1f}")

if __name__ == "__main__":
    main()
