"""t048 LocateAnything-3B on ScreenSpot-Pro (authors claim 60.3 avg).
GUI pointing: "Point to: {instruction}." -> predicted point inside GT bbox = hit.
Greedy (temp 0), generation_mode hybrid (their recommendation). Raw outputs to JSONL.

  ~/la-env/bin/python scripts/la_screenspot.py --data ~/screenspot-pro --limit 5   # sanity
  ~/la-env/bin/python scripts/la_screenspot.py --data ~/screenspot-pro            # full
"""
import argparse, glob, json, os, time
import torch
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
from la_probe import load, predict, parse_points, parse_boxes

MAX_SIDE = 5120  # in_token_limit guard; resize above this, scale GT accordingly


def iter_items(data_dir):
    for ann in sorted(glob.glob(os.path.join(data_dir, "annotations", "*.json"))):
        app = os.path.splitext(os.path.basename(ann))[0]
        for item in json.load(open(ann)):
            yield app, item


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/screenspot-pro"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="results/la-screenspot.jsonl")
    ap.add_argument("--token-limit", type=int, default=0,
                    help="override processor in_token_limit (32GB-card guard; "
                         "shipped default 25600 OOMs MoonViT's SDPA fallback)")
    args = ap.parse_args()

    tok, proc, model = load("nvidia/LocateAnything-3B")
    if args.token_limit:
        proc.image_processor.in_token_limit = args.token_limit
        print(f"in_token_limit -> {args.token_limit}", flush=True)
    items = list(iter_items(args.data))
    if args.limit:
        items = items[:: max(1, len(items) // args.limit)][:args.limit]
    print(f"{len(items)} items", flush=True)

    hits, n, t0 = 0, 0, time.time()
    per_app = {}
    with open(args.out, "w") as f:
        for app, item in items:
            img_path = os.path.join(args.data, "images", item["img_filename"])
            image = Image.open(img_path).convert("RGB")
            w, h = image.size
            scale = 1.0
            if max(w, h) > MAX_SIDE:
                scale = MAX_SIDE / max(w, h)
                image = image.resize((int(w * scale), int(h * scale)))
                w, h = image.size
            bbox = [v * scale for v in item["bbox"]]  # xyxy pixels
            instr = item["instruction"]
            try:
                answer, dt, _ = predict(tok, proc, model, image,
                                        f"Point to: {instr}.", max_new_tokens=256)
            except torch.cuda.OutOfMemoryError:
                import traceback
                if not getattr(main, "_oom_shown", False):
                    traceback.print_exc(limit=3)
                    main._oom_shown = True
                torch.cuda.empty_cache()
                f.write(json.dumps({"app": app, "img": item["img_filename"],
                                    "error": "oom"}) + "\n")
                continue
            pts = parse_points(answer, w, h)
            if not pts:
                boxes = parse_boxes(answer, w, h)
                pts = [{"x": (b["x1"] + b["x2"]) / 2, "y": (b["y1"] + b["y2"]) / 2}
                       for b in boxes[:1]]
            ok = bool(pts) and (bbox[0] <= pts[0]["x"] <= bbox[2]
                                and bbox[1] <= pts[0]["y"] <= bbox[3])
            hits += ok
            n += 1
            a = per_app.setdefault(app, [0, 0])
            a[0] += ok
            a[1] += 1
            f.write(json.dumps({"app": app, "img": item["img_filename"], "instr": instr,
                                "bbox": bbox, "pt": pts[0] if pts else None, "ok": ok,
                                "ui_type": item.get("ui_type"), "s": round(dt, 2),
                                "raw": answer[:200]}) + "\n")
            if args.limit:
                print(f"[sanity] {app} | {instr[:50]} | img {w}x{h} | bbox {bbox} | "
                      f"pt {pts[0] if pts else None} | ok={ok}", flush=True)
            elif n % 50 == 0:
                print(f"{n}/{len(items)}  acc={hits/n:.3f}  ({(time.time()-t0)/n:.1f}s/item)",
                      flush=True)

    summary = {"n": n, "acc": round(hits / max(n, 1), 4),
               "per_app": {k: round(v[0] / v[1], 3) for k, v in sorted(per_app.items())},
               "minutes": round((time.time() - t0) / 60, 1),
               "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
