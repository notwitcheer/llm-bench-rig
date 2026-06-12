"""t048 LocateAnything-3B probe: visual smoke (boxes drawn on images) + the PBD
generation-mode speed A/B (slow=AR vs fast=MTP vs hybrid) on SDPA/sm_120.
Runs on capsule in ~/la-env (transformers==4.57.1, peft) — NOT the repo .venv.

  ~/la-env/bin/python scripts/la_probe.py smoke --images ~/la-imgs
  ~/la-env/bin/python scripts/la_probe.py speed --images ~/la-imgs
"""
import argparse, glob, json, os, re, time
import torch
from PIL import Image, ImageDraw
from transformers import AutoModel, AutoTokenizer, AutoProcessor

MODEL = "nvidia/LocateAnything-3B"


def load(model_path):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    proc = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to("cuda").eval()
    return tok, proc, model


@torch.no_grad()
def predict(tok, proc, model, image, question, generation_mode="hybrid",
            max_new_tokens=2048, temperature=0.0):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": question},
    ]}]
    text = proc.py_apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos = proc.process_vision_info(messages)
    inputs = proc(text=[text], images=images, videos=videos, return_tensors="pt").to("cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    response = model.generate(
        pixel_values=inputs["pixel_values"].to(torch.bfloat16),
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        image_grid_hws=inputs.get("image_grid_hws", None),
        tokenizer=tok,
        max_new_tokens=max_new_tokens,
        use_cache=True,
        generation_mode=generation_mode,
        temperature=temperature,
        do_sample=temperature > 0,
        top_p=0.9,
        repetition_penalty=1.1,
        verbose=True,
    )
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    answer = response[0] if isinstance(response, tuple) else response
    stats = response[2] if isinstance(response, tuple) and len(response) >= 3 else None
    return answer, dt, stats


def parse_boxes(answer, w, h):
    return [{"x1": int(m[0]) / 1000 * w, "y1": int(m[1]) / 1000 * h,
             "x2": int(m[2]) / 1000 * w, "y2": int(m[3]) / 1000 * h}
            for m in re.findall(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer)]


def parse_points(answer, w, h):
    return [{"x": int(m[0]) / 1000 * w, "y": int(m[1]) / 1000 * h}
            for m in re.findall(r"<box><(\d+)><(\d+)></box>", answer)]


def draw(image, boxes, points, out_path):
    img = image.copy()
    d = ImageDraw.Draw(img)
    for b in boxes:
        d.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline="#e06060", width=4)
    for p in points:
        r = 10
        d.ellipse([p["x"] - r, p["y"] - r, p["x"] + r, p["y"] + r],
                  outline="#e8c44a", width=4)
    img.save(out_path)


def smoke(tok, proc, model, img_dir):
    os.makedirs(os.path.expanduser("~/la-out"), exist_ok=True)
    tasks = [
        ("detect", "Locate all the instances that matches the following description: "
                   "person</c>car</c>cat</c>dog</c>chair."),
        ("ground", "Locate a single instance that matches the following description: "
                   "the leftmost animal."),
        ("point", "Point to: the largest object in the scene."),
    ]
    for path in sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[:2]:
        image = Image.open(path).convert("RGB")
        w, h = image.size
        name = os.path.splitext(os.path.basename(path))[0]
        print(f"\n{'='*70}\nIMAGE {name} ({w}x{h})")
        for tag, q in tasks:
            answer, dt, _ = predict(tok, proc, model, image, q)
            boxes, points = parse_boxes(answer, w, h), parse_points(answer, w, h)
            print(f"--- [{tag}] {dt:.1f}s  {len(boxes)} boxes, {len(points)} points")
            print((answer[:300]) if answer else "(empty)")
            out = os.path.expanduser(f"~/la-out/{name}-{tag}.png")
            draw(image, boxes, points, out)
            print(f"    drew {out}")
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"peak VRAM so far: {peak:.2f} GB")


def speed(tok, proc, model, img_dir):
    path = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[0]
    image = Image.open(path).convert("RGB")
    w, h = image.size
    q = ("Locate all the instances that matches the following description: "
         "person</c>car</c>cat</c>dog</c>chair</c>bottle</c>cup.")
    predict(tok, proc, model, image, q, generation_mode="slow")  # warmup
    results = {}
    for mode in ("slow", "fast", "hybrid"):
        runs = []
        for _ in range(3):
            answer, dt, stats = predict(tok, proc, model, image, q, generation_mode=mode)
            n_boxes = len(parse_boxes(answer, w, h))
            runs.append({"s": round(dt, 2), "boxes": n_boxes,
                         "boxes_per_s": round(n_boxes / dt, 2)})
            if stats:
                runs[-1]["stats"] = stats
        results[mode] = runs
        print(f"[{mode}] {json.dumps(runs)}", flush=True)
    best = {m: max(r["boxes_per_s"] for r in rs) for m, rs in results.items()}
    if best.get("slow"):
        print(json.dumps({"speedup_fast_vs_slow": round(best["fast"] / best["slow"], 2),
                          "speedup_hybrid_vs_slow": round(best["hybrid"] / best["slow"], 2),
                          "best_boxes_per_s": best}, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["smoke", "speed", "all"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--images", default=os.path.expanduser("~/la-imgs"))
    args = ap.parse_args()
    tok, proc, model = load(args.model)
    if args.task in ("smoke", "all"):
        smoke(tok, proc, model, args.images)
    if args.task in ("speed", "all"):
        speed(tok, proc, model, args.images)
