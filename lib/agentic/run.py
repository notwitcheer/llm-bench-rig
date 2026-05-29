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
    port = get("llama_cpp.server_port", 8090)
    api_base = f"http://127.0.0.1:{port}/v1"
    base_ctx = get("agentic.base_ctx", 16384)

    print(f"\n===== {slug} =====", flush=True)
    evals = {}

    # Phase 1: shared moderate-ctx server for codeact / multistep / instruction
    # (small prompts — no need for a huge context window).
    proc = start_llama_server(path, ctx_size=base_ctx)
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
    finally:
        stop_llama_server(proc)

    # Phase 2: longcontext — its own server per depth so a depth whose KV cache won't
    # fit in VRAM scores 0 at that depth instead of failing the model's whole run.
    evals["longcontext"] = _run_longcontext(path, sp, data_dir, results_dir, cfg, api_base)

    summary = build_summary(slug, evals, get("agentic.weights"))
    (results_dir / "agentic.json").write_text(json.dumps(summary, indent=2))
    print(f"[{slug}] Hermes Pairing Score = {summary['pairing_score']}", flush=True)
    return summary

def _run_longcontext(path, sp, data_dir, results_dir, cfg, api_base):
    """Run the long-context eval one depth at a time, each under its own server sized
    for that depth. A depth whose server fails to start (OOM) scores 0 for that depth,
    leaving the model's other eval scores intact."""
    lc = cfg["longcontext"]
    margin = 2048
    by_depth = {}
    for depth in lc["depths"]:
        label = f"{depth // 1024}K"
        need = depth + lc["max_tokens"] + margin
        try:
            proc = start_llama_server(path, ctx_size=need)
        except Exception as e:
            print(f"[longcontext] {label}: server start failed ({type(e).__name__}) -> 0 (OOM)", flush=True)
            by_depth[label] = 0.0
            continue
        try:
            with LLMClient(api_base, Path(path).stem, think=lc["think"], timeout=300) as c:
                r = LongContextUseEval(c, sp, data_dir, depths=[depth], limit=lc["limit"],
                    results_dir=results_dir, max_tokens=lc["max_tokens"]).evaluate()
            by_depth.update(r.get("by_depth", {}))
        finally:
            stop_llama_server(proc)
    overall = round(sum(by_depth.values()) / len(by_depth), 2) if by_depth else 0
    result = {"score": overall, "metric": "retrieve_and_use", "by_depth": by_depth}
    if results_dir:
        (Path(results_dir) / "longcontext_detail.json").write_text(json.dumps(result, indent=2))
    return result


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
