#!/usr/bin/env python3
import argparse
import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import uvicorn

from lib.config import load_config, get
from lib.gpu import get_gpu_stats

app = FastAPI()
RESULTS_DIR = None

def find_active_progress() -> dict | None:
    if not RESULTS_DIR or not RESULTS_DIR.exists():
        return None
    for progress_file in RESULTS_DIR.rglob("progress.json"):
        try:
            data = json.loads(progress_file.read_text())
            if data.get("step") not in ("done", "error", None):
                return data
        except (json.JSONDecodeError, OSError):
            continue
    for progress_file in sorted(RESULTS_DIR.rglob("progress.json"), reverse=True):
        try:
            return json.loads(progress_file.read_text())
        except (json.JSONDecodeError, OSError):
            continue
    return None

def load_queue_state() -> list:
    queue_file = RESULTS_DIR.parent / "queue.json" if RESULTS_DIR else None
    if queue_file and queue_file.exists():
        try:
            return json.loads(queue_file.read_text())
        except (json.JSONDecodeError, OSError):
            return []
    return []

@app.get("/api/status")
def api_status():
    progress = find_active_progress()
    try:
        gpu = get_gpu_stats()
    except Exception:
        gpu = {}
    queue = load_queue_state()
    return {"progress": progress, "gpu": gpu, "queue": queue}

@app.get("/api/stream")
async def api_stream():
    async def event_generator():
        while True:
            progress = find_active_progress()
            try:
                gpu = get_gpu_stats()
            except Exception:
                gpu = {}
            queue = load_queue_state()
            payload = json.dumps({"progress": progress, "gpu": gpu, "queue": queue})
            yield f"data: {payload}\n\n"
            await asyncio.sleep(get("dashboard.gpu_poll_interval", 2))
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/results/{model_slug}")
def api_results(model_slug: str):
    model_dir = RESULTS_DIR / model_slug
    result = {}
    for name in ("meta", "speed", "quality"):
        f = model_dir / f"{name}.json"
        if f.exists():
            result[name] = json.loads(f.read_text())
    return result

@app.get("/", response_class=HTMLResponse)
def index():
    html_path = Path(__file__).parent / "templates" / "dashboard.html"
    return html_path.read_text()

def main():
    parser = argparse.ArgumentParser(description="Benchmark dashboard")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    load_config()
    global RESULTS_DIR
    RESULTS_DIR = Path(get("results_dir", "./results")).resolve()
    port = args.port or get("dashboard.port", 8085)

    print(f"Dashboard: http://0.0.0.0:{port}")
    print(f"Results dir: {RESULTS_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
