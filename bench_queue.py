#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

from lib.config import load_config, get
from lib.meta import extract_metadata

QUEUE_FILE = Path("queue.json")

def load_queue() -> list:
    if QUEUE_FILE.exists():
        return json.loads(QUEUE_FILE.read_text())
    return []

def save_queue(queue: list):
    with open(QUEUE_FILE, "w") as f:
        json.dump(queue, f, indent=2)

def cmd_add(model_path: str):
    queue = load_queue()
    path = str(Path(model_path).expanduser().resolve())
    if any(q["path"] == path for q in queue):
        print(f"Already in queue: {path}")
        return
    meta = extract_metadata(Path(path))
    queue.append({"path": path, "name": meta["name"], "slug": meta["slug"], "status": "pending"})
    save_queue(queue)
    print(f"Added: {meta['name']} ({meta['slug']})")

def cmd_add_all():
    models_dir = Path(get("models_dir", "~/.cache/huggingface/hub")).expanduser()
    count = 0
    for gguf in models_dir.rglob("*.gguf"):
        if ".no_exist" in str(gguf):
            continue
        cmd_add(str(gguf))
        count += 1
    print(f"\nScanned {count} GGUF models.")

def cmd_list():
    queue = load_queue()
    if not queue:
        print("Queue is empty.")
        return
    print(f"\n{'Status':<10} {'Name':<40} {'Slug'}")
    print("-" * 80)
    for q in queue:
        print(f"{q['status']:<10} {q['name']:<40} {q['slug']}")
    print()

def cmd_start():
    from bench import run_benchmark
    from export import export_report

    queue = load_queue()
    pending = [q for q in queue if q["status"] == "pending"]

    if not pending:
        print("No pending models in queue.")
        return

    for item in pending:
        item["status"] = "running"
        save_queue(queue)

        print(f"\n{'='*60}")
        print(f"  Starting: {item['name']}")
        print(f"{'='*60}")

        try:
            run_benchmark(item["path"])
            export_report(item["slug"])
            item["status"] = "done"
        except Exception as e:
            print(f"\nFAILED: {e}", file=sys.stderr)
            item["status"] = "error"

        save_queue(queue)
        _print_summary(item)

        if item["status"] == "done":
            action = input("\nPress Enter to continue, 'skip' to discard results, 'stop' to halt: ").strip()
            if action == "skip":
                item["status"] = "skipped"
                save_queue(queue)
            elif action == "stop":
                break

def _print_summary(item: dict):
    results_dir = Path(get("results_dir", "./results")) / item["slug"]
    print(f"\n{'─'*60}")
    print(f"  Results: {item['name']} [{item['status']}]")
    print(f"{'─'*60}")

    speed_file = results_dir / "speed.json"
    if speed_file.exists():
        speed = json.loads(speed_file.read_text())
        for k in ["pp128", "pp512", "pp2048", "tg128"]:
            if k in speed:
                print(f"  {k}: {speed[k]['tokens_per_sec']:,.1f} t/s")

    quality_file = results_dir / "quality.json"
    if quality_file.exists():
        quality = json.loads(quality_file.read_text())
        for task, result in quality.items():
            print(f"  {task}: {result['score']}%")

    report = results_dir / "report.html"
    if report.exists():
        print(f"\n  Report: {report}")
    cards = results_dir / "cards"
    if cards.exists():
        for png in cards.glob("*.png"):
            print(f"  Card:   {png}")

def cmd_results():
    load_config()
    queue = load_queue()
    done = [q for q in queue if q["status"] == "done"]
    if not done:
        print("No completed benchmarks yet.")
        return
    for item in done:
        _print_summary(item)

def cmd_skip():
    queue = load_queue()
    running = [q for q in queue if q["status"] == "running"]
    if running:
        running[0]["status"] = "skipped"
        save_queue(queue)
        print(f"Skipped: {running[0]['name']}")
    else:
        print("Nothing currently running.")

def main():
    load_config()
    parser = argparse.ArgumentParser(description="Benchmark queue manager")
    sub = parser.add_subparsers(dest="command")

    add_p = sub.add_parser("add", help="Add a model to the queue")
    add_p.add_argument("model_path")

    sub.add_parser("add-all", help="Scan HuggingFace cache and add all GGUF models")
    sub.add_parser("list", help="Show the queue")
    sub.add_parser("start", help="Start processing the queue")
    sub.add_parser("skip", help="Skip the current model")
    sub.add_parser("results", help="Show completed benchmark summaries")

    args = parser.parse_args()
    if args.command == "add":
        cmd_add(args.model_path)
    elif args.command == "add-all":
        cmd_add_all()
    elif args.command == "list":
        cmd_list()
    elif args.command == "start":
        cmd_start()
    elif args.command == "skip":
        cmd_skip()
    elif args.command == "results":
        cmd_results()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
