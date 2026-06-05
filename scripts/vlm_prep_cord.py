#!/usr/bin/env python3
"""Prepare a CORD-v2 receipt subset for the VLM extraction bench.

Writes raw PNG bytes (no Pillow needed — llama.cpp decodes images itself) plus a
ground-truth JSONL with the money fields we score against. Public, reproducible.

Usage: vlm_prep_cord.py <out_dir> <n>
"""
import json
import re
import sys
from pathlib import Path

from datasets import load_dataset, Image


def digits(v):
    """Normalize a money string to digits-only for robust comparison ('60.000' -> '60000')."""
    if v is None:
        return None
    d = re.sub(r"\D", "", str(v))
    return d.lstrip("0") or ("0" if d else None)


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "~/vlm-bench/data").expanduser()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 50
    img_dir = out / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("naver-clova-ix/cord-v2", split="test", streaming=True)
    ds = ds.cast_column("image", Image(decode=False))

    gt_path = out / "ground_truth.jsonl"
    written = 0
    with gt_path.open("w") as fh:
        for ex in ds:
            if written >= n:
                break
            raw = ex["image"]["bytes"]
            iid = f"cord_{written:03d}"
            (img_dir / f"{iid}.png").write_bytes(raw)
            gp = json.loads(ex["ground_truth"]).get("gt_parse", {})
            total = (gp.get("total") or {})
            sub = (gp.get("sub_total") or {})
            menu = gp.get("menu")
            n_items = len(menu) if isinstance(menu, list) else (1 if menu else 0)
            fh.write(json.dumps({
                "id": iid,
                "total_price": digits(total.get("total_price")),
                "subtotal_price": digits(sub.get("subtotal_price")),
                "tax_price": digits(sub.get("tax_price")),
                "line_item_count": n_items,
            }) + "\n")
            written += 1
    print(f"wrote {written} images -> {img_dir}")
    print(f"ground truth -> {gt_path}")


if __name__ == "__main__":
    main()
