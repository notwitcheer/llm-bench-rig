"""
Report generation module.

Reads benchmark result JSONs and produces:
  - Markdown report (<slug>.md)
  - Filled HTML card (<slug>-card.html)
  - X/Twitter post (<slug>-x-post.txt)
  - Telegram Grimoire post (<slug>-telegram.txt)
  - PNG card screenshot (<slug>-card.png, if Playwright available)

Usage:
    from lib.report import generate_report
    generate_report("qwen3-6-27b-q4-k-m")
"""

import json
from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from lib.config import load_config, get


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_results(model_dir: Path) -> dict:
    """Read meta, speed, quality, and optional mmlu_detail JSONs."""
    data = {}
    for name in ("meta", "speed", "quality", "mmlu_detail"):
        f = model_dir / f"{name}.json"
        if f.exists():
            with open(f) as fh:
                data[name] = json.load(fh)
    return data


def _infer_arch(meta: dict, speed: dict) -> str:
    """Return architecture label based on model name and parameter ratios."""
    name = meta.get("name", "")
    if "A1B" in name:
        return "MoE (1B active)"
    if "A3B" in name:
        return "MoE (3B active)"

    # Heuristic: large param count in small file usually means MoE
    params_str = speed.get("params", "")
    if params_str:
        try:
            params_b = float(params_str.replace(" B", ""))
            size_gib = speed.get("model_size_gib", 0)
            if params_b > 50 and size_gib < 30:
                return "MoE"
        except ValueError:
            pass

    if "MTP" in name:
        return "Dense + MTP"
    return "Dense"


def _fmt_speed(val: float) -> str:
    """Format a speed value with comma separators and one decimal."""
    if val >= 1000:
        return f"{val:,.0f}"
    return f"{val:.1f}"


def _build_context(data: dict, slug: str = "") -> dict:
    """Assemble the full template context dict from loaded result data."""
    meta = data.get("meta", {})
    speed = data.get("speed", {})
    quality = data.get("quality", {})
    mmlu_detail = data.get("mmlu_detail", None)
    hardware = get("hardware", {})
    config = load_config()

    arch = _infer_arch(meta, speed)
    params = speed.get("params", "?")
    size_gib = meta.get("size_gib", speed.get("model_size_gib", "?"))
    quant = meta.get("quant", "?")
    engine = meta.get("engine", speed.get("engine", "llama.cpp"))

    # Speed keys sorted
    pp_keys = sorted(
        [k for k in speed if k.startswith("pp") and isinstance(speed[k], dict)],
        key=lambda k: int(k[2:]),
    )
    tg_keys = sorted(
        [k for k in speed if k.startswith("tg") and isinstance(speed[k], dict)],
        key=lambda k: int(k[2:]),
    )

    # Key speed values for quick access
    pp512 = speed.get("pp512", {}).get("tokens_per_sec", 0)
    tg128 = speed.get("tg128", {}).get("tokens_per_sec", 0)

    # Quality sampling info
    sample_ratio = config.get("quality", {}).get("sample")
    think_enabled = config.get("quality", {}).get("think", False)

    ctx = {
        # Identity
        "model_name": meta.get("name", meta.get("slug", "Unknown")),
        "slug": meta.get("slug", ""),
        "arch": arch,
        "params": params,
        "quant": quant,
        "size_gib": size_gib,
        "engine": engine,
        "backend": speed.get("backend", "CUDA"),
        "date": date.today().isoformat(),

        # Hardware
        "hardware": hardware,
        "gpu": hardware.get("gpu", "Unknown GPU"),
        "vram": hardware.get("vram", "?"),
        "cpu": hardware.get("cpu", "?"),
        "ram": hardware.get("ram", "?"),
        "os": hardware.get("os", "?"),
        "cuda": hardware.get("cuda", "?"),

        # Raw data
        "meta": meta,
        "speed": speed,
        "quality": quality,
        "mmlu_detail": mmlu_detail,

        # Speed helpers
        "pp_keys": pp_keys,
        "tg_keys": tg_keys,
        "pp512": pp512,
        "tg128": tg128,
        "pp512_fmt": _fmt_speed(pp512),
        "tg128_fmt": _fmt_speed(tg128),

        # Quality helpers
        "has_quality": bool(quality),
        "has_mmlu_detail": mmlu_detail is not None,

        # Methodology
        "sample_ratio": sample_ratio,
        "think_enabled": think_enabled,

        # Links
        "github_url": f"https://github.com/notwitcheer/llm-bench-rig/blob/main/reports/{slug}.md",
        "hf_url": f"https://huggingface.co/datasets/witcheer/rtx-5090-benchmarks/blob/main/reports/{slug}.md",
    }
    return ctx


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------

def generate_markdown(ctx: dict) -> str:
    """Generate a full markdown benchmark report."""
    lines = []
    w = lines.append

    name = ctx["model_name"]
    quant = ctx["quant"]
    slug = ctx["slug"]

    w(f"# Benchmark Report: {name} ({quant})")
    w("")
    w(f"**Date:** {ctx['date']}  ")
    w("**Author:** WITCHEER  ")
    w(f"**Platform:** {ctx['gpu']} Benchmark Rig (capsule)  ")
    w("")
    w("---")
    w("")

    # -- Model table --
    w("## Model")
    w("")
    w("| Field | Value |")
    w("|-------|-------|")
    w(f"| Model | {name} |")
    arch_note = f" ({ctx['arch'].lower()})" if ctx["arch"] != "Dense" else " (dense)"
    w(f"| Parameters | {ctx['params']}{arch_note} |")
    w(f"| Quantization | {quant} |")
    w(f"| File size | {ctx['size_gib']} GiB |")
    w(f"| Engine | {ctx['engine']} (CUDA {ctx['cuda']}) |")
    w("")

    # -- Hardware table --
    w("## Hardware")
    w("")
    w("| Component | Spec |")
    w("|-----------|------|")
    w(f"| GPU | {ctx['gpu']} |")
    w(f"| CPU | {ctx['cpu']} |")
    w(f"| RAM | {ctx['ram']} |")
    w(f"| OS | {ctx['os']} |")
    w(f"| CUDA | {ctx['cuda']} |")
    w("")
    w("---")
    w("")

    # -- Quality benchmarks --
    if ctx["has_quality"]:
        quality = ctx["quality"]
        w("## Quality Benchmarks")
        w("")
        w("All benchmarks use generative evaluation via llama-server chat completions. "
          "Multiple-choice tasks (MMLU, ARC, HellaSwag) use letter extraction instead of "
          "loglikelihood scoring -- results are internally consistent for model comparison "
          "but absolute scores may differ from logprob-based evaluations by 5-15%.")
        w("")

        w("### Summary")
        w("")
        w("| Benchmark | Score | Metric |")
        w("|-----------|------:|--------|")
        bench_display = {
            "mmlu": "MMLU",
            "arc_challenge": "ARC-Challenge",
            "hellaswag": "HellaSwag",
            "humaneval": "HumanEval",
            "gsm8k": "GSM8K",
        }
        for key, vals in quality.items():
            display = bench_display.get(key, key)
            score = vals.get("score", 0)
            metric = vals.get("metric", "acc")
            if metric in ("acc", "accuracy"):
                w(f"| **{display}** | **{score:.2f}%** | accuracy |")
            elif metric == "pass@1":
                w(f"| **{display}** | **{score:.2f}%** | pass@1 |")
            else:
                w(f"| **{display}** | **{score:.2f}%** | {metric} |")
        w("")

        # -- MMLU breakdown --
        if ctx["has_mmlu_detail"]:
            mmlu = ctx["mmlu_detail"]
            categories = mmlu.get("categories", {})
            if categories:
                w("### MMLU Breakdown by Category")
                w("")
                w("| Category | Score | Correct / Total |")
                w("|----------|------:|----------------:|")
                for cat_name, cat_data in categories.items():
                    cat_display = cat_name.replace("_", " ").title()
                    cat_score = cat_data.get("score", 0)
                    correct = cat_data.get("correct", "?")
                    total = cat_data.get("total", "?")
                    w(f"| {cat_display} | {cat_score:.2f}% | {correct:,} / {total:,} |")
                w("")
                if mmlu.get("sample"):
                    w(f"*Sampled at {mmlu['sample'] * 100:.0f}% (seed {mmlu.get('sample_seed', '?')})*")
                    w("")

        w("---")
        w("")

    # -- Speed benchmarks --
    w("## Speed Benchmarks")
    w("")
    w("Measured with `llama-bench`. All layers GPU-offloaded (`-ngl 99`).")
    w("")

    if ctx["pp_keys"]:
        w("### Prompt Processing (tokens/s)")
        w("")
        w("| Context Length | Speed | +/-sigma |")
        w("|---------------:|------:|---------:|")
        for k in ctx["pp_keys"]:
            length = k[2:]
            tps = ctx["speed"][k]["tokens_per_sec"]
            std = ctx["speed"][k].get("stddev", 0)
            w(f"| {length} | {_fmt_speed(tps)} | {std:.1f} |")
        w("")

    if ctx["tg_keys"]:
        w("### Generation (tokens/s)")
        w("")
        w("| Metric | Speed | +/-sigma |")
        w("|--------|------:|---------:|")
        for k in ctx["tg_keys"]:
            tps = ctx["speed"][k]["tokens_per_sec"]
            std = ctx["speed"][k].get("stddev", 0)
            w(f"| {k} | {_fmt_speed(tps)} | {std:.1f} |")
        w("")

    w("---")
    w("")

    # -- Methodology --
    w("## Methodology")
    w("")
    w("### Evaluation Framework")
    w("")
    w("Custom generative evaluators built for this rig. All benchmarks run through "
      "llama-server's `/v1/chat/completions` endpoint.")
    w("")
    w("- **Scoring:** Generative evaluation (not loglikelihood)")
    think_note = "enabled" if ctx["think_enabled"] else "disabled"
    w(f"- **Thinking:** {think_note}")
    w("- **MCQ scoring:** First valid letter extracted from response (A/B/C/D)")
    if ctx["sample_ratio"] and ctx["sample_ratio"] < 1.0:
        w(f"- **Sampling:** {ctx['sample_ratio'] * 100:.0f}% of dataset used")
    w("- **Temperature:** 0 (deterministic)")
    w("- **Max tokens:** 2,048")
    w("- **GPU offload:** All layers (`-ngl 99`)")
    w("")

    w("---")
    w("")
    w(f"*Benchmarked by WITCHEER on the RTX 5090 Benchmark Rig. "
      f"Source: [{ctx['github_url'].replace('https://', '')}]({ctx['github_url']}). "
      f"Dataset: [{ctx['hf_url'].replace('https://', '')}]({ctx['hf_url']}).*")
    w("")

    return "\n".join(lines)


def generate_card_html(ctx: dict) -> str:
    """Render the card-model-report.html jinja2 template."""
    env = Environment(loader=FileSystemLoader("templates"))
    tmpl = env.get_template("card-model-report.html")
    return tmpl.render(**ctx)


def generate_x_post(ctx: dict) -> str:
    """Generate a ~280 character X/Twitter post."""
    name = ctx["model_name"]
    quant = ctx["quant"]

    # Quality scores line
    quality_parts = []
    quality = ctx.get("quality", {})
    if "mmlu" in quality:
        quality_parts.append(f"MMLU {quality['mmlu']['score']:.1f}%")
    if "arc_challenge" in quality:
        quality_parts.append(f"ARC {quality['arc_challenge']['score']:.1f}%")
    if "gsm8k" in quality:
        quality_parts.append(f"GSM8K {quality['gsm8k']['score']:.1f}%")
    if "hellaswag" in quality:
        quality_parts.append(f"HellaSwag {quality['hellaswag']['score']:.1f}%")
    if "humaneval" in quality:
        quality_parts.append(f"HumanEval {quality['humaneval']['score']:.1f}%")

    lines = []
    lines.append(f"{name} ({quant}) on RTX 5090")
    lines.append("")
    if quality_parts:
        # Split into two lines if many scores
        mid = (len(quality_parts) + 1) // 2
        lines.append(" | ".join(quality_parts[:mid]))
        if quality_parts[mid:]:
            lines.append(" | ".join(quality_parts[mid:]))
        lines.append("")

    speed_parts = []
    if ctx["tg128"]:
        speed_parts.append(f"tg128: {ctx['tg128_fmt']} t/s")
    if ctx["pp512"]:
        speed_parts.append(f"pp512: {ctx['pp512_fmt']} t/s")
    if speed_parts:
        lines.append(" | ".join(speed_parts))

    lines.append(f"VRAM: {ctx['size_gib']} GiB ({quant})")
    lines.append("")
    lines.append(ctx["github_url"])

    post = "\n".join(lines)

    # Trim to ~280 chars if needed by dropping less important lines
    if len(post) > 280:
        # Remove HumanEval line, hellaswag if still too long
        quality_parts_short = [p for p in quality_parts
                               if not p.startswith("HumanEval") and not p.startswith("HellaSwag")]
        lines_short = [f"{name} ({quant}) on RTX 5090", ""]
        if quality_parts_short:
            lines_short.append(" | ".join(quality_parts_short))
            lines_short.append("")
        if speed_parts:
            lines_short.append(" | ".join(speed_parts))
        lines_short.append(f"VRAM: {ctx['size_gib']} GiB")
        lines_short.append("")
        lines_short.append(ctx["github_url"])
        post = "\n".join(lines_short)

    return post


def generate_telegram(ctx: dict) -> str:
    """Generate a ~200 word Telegram Grimoire post."""
    name = ctx["model_name"]
    quant = ctx["quant"]
    arch = ctx["arch"]
    quality = ctx.get("quality", {})

    lines = []

    # Bold title
    lines.append(f"<b>{name} ({quant}) -- RTX 5090 Benchmark</b>")
    lines.append("")

    # Verdict paragraph
    verdict_parts = []
    if "mmlu" in quality:
        mmlu_score = quality["mmlu"]["score"]
        if mmlu_score >= 85:
            verdict_parts.append(f"strong knowledge coverage (MMLU {mmlu_score:.1f}%)")
        elif mmlu_score >= 70:
            verdict_parts.append(f"solid knowledge coverage (MMLU {mmlu_score:.1f}%)")
        else:
            verdict_parts.append(f"moderate knowledge coverage (MMLU {mmlu_score:.1f}%)")
    if "gsm8k" in quality:
        gsm = quality["gsm8k"]["score"]
        if gsm >= 90:
            verdict_parts.append(f"excellent math reasoning (GSM8K {gsm:.1f}%)")
        else:
            verdict_parts.append(f"GSM8K {gsm:.1f}%")
    if "arc_challenge" in quality:
        arc = quality["arc_challenge"]["score"]
        verdict_parts.append(f"ARC-Challenge {arc:.1f}%")
    if "humaneval" in quality:
        he = quality["humaneval"]["score"]
        verdict_parts.append(f"HumanEval {he:.1f}%")

    arch_desc = f"{arch} architecture" if arch != "Dense" else "dense architecture"
    lines.append(f"Benchmarked {name} ({ctx['params']}, {arch_desc}, {quant} quantization) "
                 f"on the RTX 5090 capsule rig.")
    if verdict_parts:
        lines.append(f"Results: {', '.join(verdict_parts)}.")
    lines.append("")

    # Speed context paragraph
    speed_lines = []
    if ctx["tg128"]:
        speed_lines.append(f"Generation speed: {ctx['tg128_fmt']} tokens/s (tg128)")
    if ctx["pp512"]:
        speed_lines.append(f"prompt processing: {ctx['pp512_fmt']} tokens/s (pp512)")
    speed_lines.append(f"Model fits in {ctx['size_gib']} GiB VRAM at {quant}.")

    lines.append(" ".join(speed_lines))

    # Methodology note
    think_note = "enabled" if ctx["think_enabled"] else "disabled"
    method_parts = [f"Generative evaluation (thinking {think_note}), "
                    f"MCQ via letter extraction, temperature 0."]
    if ctx["sample_ratio"] and ctx["sample_ratio"] < 1.0:
        method_parts.append(f" Dataset sampled at {ctx['sample_ratio'] * 100:.0f}%.")
    lines.append("".join(method_parts))
    lines.append("")

    # Link
    lines.append(f"Full results and methodology: {ctx['github_url']}")
    lines.append(f"Dataset: {ctx['hf_url']}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# PNG rendering
# ---------------------------------------------------------------------------

def _render_card_png(html_path: Path, png_path: Path):
    """Screenshot the HTML card at 1200x675 via Playwright. Graceful fallback."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright not installed -- skipping PNG export. "
              "Run: pip install playwright && playwright install chromium")
        return

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page(viewport={"width": 1200, "height": 675})
            page.goto(f"file://{html_path.resolve()}")
            page.wait_for_timeout(1000)
            page.screenshot(path=str(png_path))
            print(f"Card PNG: {png_path}")
            page.close()
            browser.close()
    except Exception as e:
        print(f"PNG rendering failed: {e}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_report(model_slug: str):
    """
    Generate all report outputs for a benchmarked model.

    Reads results from results/<model_slug>/, writes outputs to reports/:
      - <slug>.md
      - <slug>-card.html
      - <slug>-x-post.txt
      - <slug>-telegram.txt
      - <slug>-card.png  (if Playwright available)
    """
    load_config()
    results_dir = Path(get("results_dir", "./results"))
    model_dir = results_dir / model_slug

    if not model_dir.exists():
        raise FileNotFoundError(f"No results directory found: {model_dir}")

    data = _load_results(model_dir)
    if "meta" not in data:
        raise FileNotFoundError(f"meta.json not found in {model_dir}")

    ctx = _build_context(data, slug=model_slug)

    # Ensure output directory exists
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)

    generated = []

    # 1. Markdown report
    md_path = reports_dir / f"{model_slug}.md"
    md_path.write_text(generate_markdown(ctx))
    generated.append(md_path)

    # 2. HTML card (via jinja2 template)
    html_path = reports_dir / f"{model_slug}-card.html"
    try:
        html = generate_card_html(ctx)
        html_path.write_text(html)
        generated.append(html_path)
    except Exception as e:
        print(f"Warning: card HTML generation failed: {e}")
        html_path = None

    # 3. X post
    x_path = reports_dir / f"{model_slug}-x-post.txt"
    x_path.write_text(generate_x_post(ctx))
    generated.append(x_path)

    # 4. Telegram post
    tg_path = reports_dir / f"{model_slug}-telegram.txt"
    tg_path.write_text(generate_telegram(ctx))
    generated.append(tg_path)

    # 5. PNG rendering (best-effort)
    if html_path and html_path.exists():
        png_path = reports_dir / f"{model_slug}-card.png"
        _render_card_png(html_path, png_path)
        if png_path.exists():
            generated.append(png_path)

    # Print summary
    print(f"\nGenerated {len(generated)} report files for {model_slug}:")
    for p in generated:
        print(f"  {p}")

    return generated
