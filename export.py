#!/usr/bin/env python3
import argparse
import json
import sys
from datetime import date
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from lib.config import load_config, get

def load_model_data(model_dir: Path) -> dict:
    data = {}
    for name in ("meta", "speed", "quality"):
        f = model_dir / f"{name}.json"
        if f.exists():
            with open(f) as fh:
                data[name] = json.load(fh)
    return data

def build_template_context(data: dict) -> dict:
    meta = data.get("meta", {})
    speed = data.get("speed", {})
    quality = data.get("quality", {})
    hardware = get("hardware", {})

    speed_labels = []
    speed_values = []
    pp_keys = sorted(
        [k for k in speed if k.startswith("pp") and isinstance(speed[k], dict)],
        key=lambda k: int(k[2:]),
    )
    tg_keys = sorted(
        [k for k in speed if k.startswith("tg") and isinstance(speed[k], dict)],
        key=lambda k: int(k[2:]),
    )
    for k in pp_keys + tg_keys:
        speed_labels.append(k)
        speed_values.append(speed[k]["tokens_per_sec"])

    quality_labels = list(quality.keys())
    quality_values = [v["score"] for v in quality.values()]

    return {
        "meta": meta,
        "speed": speed,
        "quality": quality,
        "hardware": hardware,
        "date": date.today().isoformat(),
        "speed_labels": speed_labels,
        "speed_values": speed_values,
        "quality_labels": quality_labels,
        "quality_values": quality_values,
    }

def export_report(model_slug: str):
    load_config()
    results_dir = Path(get("results_dir", "./results"))
    model_dir = results_dir / model_slug

    if not model_dir.exists():
        print(f"No results found for {model_slug}", file=sys.stderr)
        sys.exit(1)

    data = load_model_data(model_dir)
    ctx = build_template_context(data)

    env = Environment(loader=FileSystemLoader("templates"))

    report = env.get_template("report.html").render(**ctx)
    (model_dir / "report.html").write_text(report)
    print(f"Report: {model_dir / 'report.html'}")

    cards_dir = model_dir / "cards"
    cards_dir.mkdir(exist_ok=True)

    for card_name in ["card-speed", "card-quality", "card-summary"]:
        try:
            tmpl = env.get_template(f"{card_name}.html")
            html = tmpl.render(**ctx)
            html_path = cards_dir / f"{card_name}.html"
            html_path.write_text(html)
        except Exception as e:
            print(f"Warning: {card_name} template failed: {e}")

    _render_pngs(cards_dir)

def _render_pngs(cards_dir: Path):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright not installed — skipping PNG export. Run: pip install playwright && playwright install chromium")
        return

    with sync_playwright() as p:
        browser = p.chromium.launch()
        for html_file in cards_dir.glob("*.html"):
            page = browser.new_page(viewport={"width": 1200, "height": 675})
            page.goto(f"file://{html_file.resolve()}")
            page.wait_for_timeout(1000)
            png_path = html_file.with_suffix(".png")
            page.screenshot(path=str(png_path))
            print(f"Card: {png_path}")
            page.close()
        browser.close()

def export_comparison():
    load_config()
    results_dir = Path(get("results_dir", "./results"))
    models = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        data = load_model_data(model_dir)
        if "meta" in data and "speed" in data:
            models.append(data)

    if len(models) < 2:
        print("Need at least 2 benchmarked models for comparison.", file=sys.stderr)
        sys.exit(1)

    env = Environment(loader=FileSystemLoader("templates"))
    hardware = get("hardware", {})

    # Compute sorted speed keys from first model for templates
    first_speed = models[0].get("speed", {})
    pp_keys = sorted(
        [k for k in first_speed if k.startswith("pp") and isinstance(first_speed[k], dict)],
        key=lambda k: int(k[2:]),
    )
    tg_keys = sorted(
        [k for k in first_speed if k.startswith("tg") and isinstance(first_speed[k], dict)],
        key=lambda k: int(k[2:]),
    )

    ctx = {
        "models": models,
        "hardware": hardware,
        "date": date.today().isoformat(),
        "pp_keys": pp_keys,
        "tg_keys": tg_keys,
    }

    try:
        tmpl = env.get_template("comparison.html")
        html = tmpl.render(**ctx)
        out = results_dir / "comparison.html"
        out.write_text(html)
        print(f"Comparison: {out}")
    except Exception as e:
        print(f"Comparison template error: {e}", file=sys.stderr)

    # Render comparison card (social media infographic)
    try:
        tmpl = env.get_template("card-comparison.html")
        html = tmpl.render(**ctx)
        card_html = results_dir / "card-comparison.html"
        card_html.write_text(html)
        print(f"Comparison card: {card_html}")
        _render_card_png(card_html, results_dir / "card-comparison.png")
    except Exception as e:
        print(f"Comparison card template error: {e}", file=sys.stderr)


def _infer_architecture(meta: dict, speed: dict) -> str:
    name = meta.get("name", "")
    params_str = speed.get("params", "")
    params_b = float(params_str.replace(" B", "")) if params_str else 0
    size_gib = speed.get("model_size_gib", 0)

    if "MTP" in name:
        return "Dense + MTP"
    if "A3B" in name:
        return "MoE (3B active)"
    if params_b > 50 and size_gib < 30:
        return "MoE"
    return "Dense"


def export_leaderboard():
    load_config()
    results_dir = Path(get("results_dir", "./results"))
    env = Environment(loader=FileSystemLoader("templates"))
    hardware = get("hardware", {})

    entries = []
    for model_dir in sorted(results_dir.iterdir()):
        if not model_dir.is_dir():
            continue
        data = load_model_data(model_dir)
        if "meta" not in data or "speed" not in data:
            continue
        meta, speed = data["meta"], data["speed"]
        pp512 = speed.get("pp512", {}).get("tokens_per_sec", 0)
        tg128 = speed.get("tg128", {}).get("tokens_per_sec", 0)
        if pp512 == 0 and tg128 == 0:
            continue

        params_str = speed.get("params", "")
        params_b = params_str.replace(" B", "") if params_str else "?"

        display_name = meta["name"]
        if model_dir.name == "qwen3-6-27b-mtp-q4-k-m":
            display_name = "Qwen3.6-27B-MTP"

        entries.append({
            "name": display_name,
            "arch": _infer_architecture(meta, speed),
            "params_b": params_b,
            "size_gib": speed.get("model_size_gib", meta.get("size_gib", "?")),
            "pp512": pp512,
            "tg128": tg128,
        })

    entries.sort(key=lambda e: e["tg128"], reverse=True)

    ctx = {
        "leaderboard": entries,
        "hardware": hardware,
        "date": date.today().isoformat(),
    }

    tmpl = env.get_template("card-leaderboard.html")
    html = tmpl.render(**ctx)
    card_html = results_dir / "card-leaderboard.html"
    card_html.write_text(html)
    print(f"Leaderboard card: {card_html}")
    _render_card_png(card_html, results_dir / "card-leaderboard.png")


def export_quality_leaderboard():
    """Quality leaderboard split by think mode (reasoning ON vs OFF — comparing
    across the two is invalid). Prints a ranked text table per group and renders a
    gold-crimson card. Unknown-think results are surfaced, never silently dropped."""
    from lib.board import build_quality_board, QUALITY_TASKS

    load_config()
    results_dir = Path(get("results_dir", "./results"))
    hardware = get("hardware", {})
    board = build_quality_board(results_dir)

    short = {"mmlu": "MMLU", "arc_challenge": "ARC", "hellaswag": "Hella",
             "humaneval": "HEval", "gsm8k": "GSM8K"}

    def _fmt(x):
        return f"{x:.1f}" if isinstance(x, (int, float)) else "  -  "

    for title, key in (("THINKING OFF (non-reasoning)", "off"),
                       ("THINKING ON (reasoning)", "on")):
        rows = board[key]
        print(f"\n=== {title} — {len(rows)} models ===")
        header = f"{'#':>2}  {'model':<40} {'q_avg':>6}  " + "  ".join(f"{short[t]:>6}" for t in QUALITY_TASKS)
        print(header)
        for i, e in enumerate(rows, 1):
            cells = "  ".join(f"{_fmt(e['scores'][t]):>6}" for t in QUALITY_TASKS)
            print(f"{i:>2}  {e['slug']:<40} {e['q_avg']:>6.2f}  {cells}")
    if board["unknown"]:
        print(f"\n!! UNKNOWN think mode (excluded from board): "
              + ", ".join(e["slug"] for e in board["unknown"]))

    env = Environment(loader=FileSystemLoader("templates"))
    ctx = {
        "off": board["off"],
        "on": board["on"],
        "tasks": list(QUALITY_TASKS),
        "task_labels": short,
        "hardware": hardware,
        "date": date.today().isoformat(),
    }
    tmpl = env.get_template("card-quality-leaderboard.html")
    card_html = results_dir / "card-quality-leaderboard.html"
    card_html.write_text(tmpl.render(**ctx))
    print(f"\nQuality leaderboard card: {card_html}")
    _render_card_png(card_html, results_dir / "card-quality-leaderboard.png", full_page=True)


def _render_card_png(html_path: Path, png_path: Path, full_page: bool = False):
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("Playwright not installed — skipping PNG export. Run: pip install playwright && playwright install chromium")
        return

    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 675})
        page.goto(f"file://{html_path.resolve()}")
        page.wait_for_timeout(1000)
        page.screenshot(path=str(png_path), full_page=full_page)
        print(f"Card PNG: {png_path}")
        page.close()
        browser.close()

def main():
    parser = argparse.ArgumentParser(description="Export benchmark results")
    parser.add_argument("target", help="Model slug, 'compare', 'leaderboard', or 'quality-leaderboard'")
    args = parser.parse_args()

    if args.target == "compare":
        export_comparison()
    elif args.target == "leaderboard":
        export_leaderboard()
    elif args.target in ("quality-leaderboard", "quality-board"):
        export_quality_leaderboard()
    elif args.target.startswith("report:"):
        from lib.report import generate_report
        slug = args.target.split(":", 1)[1]
        generate_report(slug)
    else:
        export_report(args.target)

if __name__ == "__main__":
    main()
