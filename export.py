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
    ctx = {"models": models, "hardware": hardware, "date": date.today().isoformat()}

    try:
        tmpl = env.get_template("comparison.html")
        html = tmpl.render(**ctx)
        out = results_dir / "comparison.html"
        out.write_text(html)
        print(f"Comparison: {out}")
    except Exception as e:
        print(f"Comparison template error: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Export benchmark results")
    parser.add_argument("target", help="Model slug to export, or 'compare' for cross-model comparison")
    args = parser.parse_args()

    if args.target == "compare":
        export_comparison()
    else:
        export_report(args.target)

if __name__ == "__main__":
    main()
