"""Score an ASR run: read results/asr/<model>/{transcripts.jsonl,timing.json}, compute
per-split micro-WER + RTFx, write results/asr/summary.json across all models. Pure scoring
on the Mac (deterministic, re-runnable) — generation happened on capsule."""
import json
import sys
from pathlib import Path
from lib.asr.score import score_split, rtfx


def summarize_model(model_dir: str) -> dict:
    p = Path(model_dir)
    rows = [json.loads(l) for l in (p / "transcripts.jsonl").read_text().splitlines() if l.strip()]
    timing = json.loads((p / "timing.json").read_text())
    clean = [(r["reference"], r["hypothesis"]) for r in rows if r["split"] == "test-clean"]
    other = [(r["reference"], r["hypothesis"]) for r in rows if r["split"] == "test-other"]
    return {
        "model": p.name,
        "wer_clean": score_split(clean)["wer"],
        "wer_other": score_split(other)["wer"],
        "rtfx": rtfx(timing["audio_seconds"], timing["proc_seconds"]),
        "peak_vram_mib": timing.get("peak_vram_mib"),
        "n_clean": len(clean), "n_other": len(other),
    }


def main(base="results/asr"):
    base = Path(base)
    summary = []
    for d in sorted(base.iterdir()):
        if d.is_dir() and (d / "transcripts.jsonl").exists():
            summary.append(summarize_model(str(d)))
    (base / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{'model':28} {'WER clean':>10} {'WER other':>10} {'RTFx':>7} {'VRAM MiB':>9}")
    for s in summary:
        print(f"{s['model']:28} {s['wer_clean']*100:9.2f}% {s['wer_other']*100:9.2f}% "
              f"{s['rtfx']:7.1f} {str(s['peak_vram_mib']):>9}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/asr")
