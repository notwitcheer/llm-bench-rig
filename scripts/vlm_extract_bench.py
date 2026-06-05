#!/usr/bin/env python3
"""VLM extraction benchmark (rig's first VLM bench).

Two pillars, mirroring the text bench (speed + quality), but for vision extraction:
  SPEED   — prompt tok/s (incl. vision tokens) + generation tok/s, from llama-server timings.
  QUALITY — JSON-validity rate + schema-consistency F1 + field-accuracy vs ground truth.

Talks to a llama-server started with --mmproj (OpenAI-compatible multimodal API).
Stores every raw model output so results can be re-scored offline with zero GPU (RUNBOOK ethos).

Usage: vlm_extract_bench.py <data_dir> <out_dir> [--url http://127.0.0.1:8091] [--limit N]
"""
import argparse
import base64
import json
import re
import statistics as st
import time
import urllib.request
from pathlib import Path

REQUESTED_KEYS = ["total_price", "subtotal_price", "tax_price", "line_item_count"]
MONEY_FIELDS = ["total_price", "subtotal_price", "tax_price"]

SYSTEM = """You extract structured data from receipt images. Return ONLY a strict JSON object with exactly these keys:
total_price: the grand total amount (digits only)
subtotal_price: the subtotal before tax/discount (digits only)
tax_price: the tax amount (digits only)
line_item_count: integer count of distinct line items"""


def digits(v):
    if v is None:
        return None
    d = re.sub(r"\D", "", str(v))
    return d.lstrip("0") or ("0" if d else None)


def extract_json(text):
    """Best-effort: strip fences, grab the first balanced {...}, parse. Returns (obj|None, raw)."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?|```$", "", t, flags=re.MULTILINE).strip()
    try:
        return json.loads(t), text
    except Exception:
        pass
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0)), text
        except Exception:
            return None, text
    return None, text


def call(url, img_path, n_predict=192):
    b64 = base64.b64encode(Path(img_path).read_bytes()).decode()
    payload = {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": [
                {"type": "text", "text": "Extract the fields as JSON."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ]},
        ],
        "temperature": 0.0,
        "n_predict": n_predict,
        "cache_prompt": False,
    }
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        resp = json.loads(r.read().decode())
    wall = time.time() - t0
    content = resp["choices"][0]["message"]["content"]
    timings = resp.get("timings", {}) or {}
    return content, timings, wall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    data = Path(args.data_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    gt = {}
    for line in (data / "ground_truth.jsonl").read_text().splitlines():
        o = json.loads(line)
        gt[o["id"]] = o
    ids = sorted(gt)
    if args.limit:
        ids = ids[:args.limit]

    rows = []
    valid = 0
    f1s, p_tps, g_tps, gen_n = [], [], [], []
    field_hits = {f: [0, 0] for f in MONEY_FIELDS}  # [correct, scorable]
    lic_hits = [0, 0]

    for i, iid in enumerate(ids):
        img = data / "images" / f"{iid}.png"
        content, timings, wall = call(args.url, img)
        obj, raw = extract_json(content)
        is_valid = obj is not None and isinstance(obj, dict)
        row = {"id": iid, "valid": is_valid, "raw": raw, "wall_s": round(wall, 3)}
        if timings:
            row["prompt_tps"] = timings.get("prompt_per_second")
            row["gen_tps"] = timings.get("predicted_per_second")
            row["prompt_n"] = timings.get("prompt_n")
            row["gen_n"] = timings.get("predicted_n")
            if row["prompt_tps"]:
                p_tps.append(row["prompt_tps"])
            if row["gen_tps"]:
                g_tps.append(row["gen_tps"])
            if row["gen_n"]:
                gen_n.append(row["gen_n"])
        if is_valid:
            valid += 1
            ret = set(obj.keys())
            req = set(REQUESTED_KEYS)
            inter = len(ret & req)
            prec = inter / len(ret) if ret else 0.0
            rec = inter / len(req)
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            f1s.append(f1)
            row["schema_f1"] = round(f1, 3)
            for f in MONEY_FIELDS:
                g = gt[iid].get(f)
                if g is None:
                    continue
                field_hits[f][1] += 1
                if digits(obj.get(f)) == g:
                    field_hits[f][0] += 1
            if gt[iid].get("line_item_count") is not None and "line_item_count" in obj:
                lic_hits[1] += 1
                try:
                    if int(obj["line_item_count"]) == int(gt[iid]["line_item_count"]):
                        lic_hits[0] += 1
                except Exception:
                    pass
        rows.append(row)
        print(f"[{i+1}/{len(ids)}] {iid} valid={is_valid} "
              f"prompt={row.get('prompt_tps')} gen={row.get('gen_tps')} tps")

    n = len(ids)

    def acc(pair):
        return round(pair[0] / pair[1], 4) if pair[1] else None

    summary = {
        "model": "LFM2.5-VL-1.6B-Extract",
        "quant": "F16 (model+mmproj)",
        "engine": "llama.cpp llama-server b9365",
        "dataset": "CORD-v2 test",
        "n_images": n,
        "quality": {
            "json_validity_rate": round(valid / n, 4) if n else None,
            "schema_consistency_f1_mean": round(st.mean(f1s), 4) if f1s else None,
            "field_accuracy": {f: {"acc": acc(field_hits[f]), "n": field_hits[f][1]} for f in MONEY_FIELDS},
            "money_field_accuracy_overall": acc([sum(field_hits[f][0] for f in MONEY_FIELDS),
                                                 sum(field_hits[f][1] for f in MONEY_FIELDS)]),
            "line_item_count_exact": {"acc": acc(lic_hits), "n": lic_hits[1]},
        },
        "speed": {
            "prompt_tps_median": round(st.median(p_tps), 1) if p_tps else None,
            "prompt_tps_mean": round(st.mean(p_tps), 1) if p_tps else None,
            "gen_tps_median": round(st.median(g_tps), 1) if g_tps else None,
            "gen_tps_mean": round(st.mean(g_tps), 1) if g_tps else None,
            "gen_tokens_mean": round(st.mean(gen_n), 1) if gen_n else None,
        },
    }

    (out / "vlm_results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {out/'vlm_results.json'}")


if __name__ == "__main__":
    main()
