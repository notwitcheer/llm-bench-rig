"""t048 diagnostic: find the largest in_token_limit that fits a 32GB card for
MoonViT's SDPA fallback (no flash-attn). One 4K screenshot, three limits.
Run DRAINED: ~/la-env/bin/python scripts/la_oomtest.py <image_path>"""
import sys, traceback
import torch
from PIL import Image
from la_probe import load, predict

img_path = sys.argv[1]
tok, proc, model = load("nvidia/LocateAnything-3B")
image = Image.open(img_path).convert("RGB")
print(f"image {image.size}", flush=True)

for limit in (25600, 16384, 12288):
    proc.image_processor.in_token_limit = limit
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        answer, dt, _ = predict(tok, proc, model, image,
                                "Point to: the main menu bar.", max_new_tokens=64)
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"limit={limit}: OK  {dt:.1f}s  peak={peak:.1f}GB  out={answer[:80]}",
              flush=True)
    except torch.cuda.OutOfMemoryError:
        peak = torch.cuda.max_memory_allocated() / 2**30
        print(f"limit={limit}: OOM  peak={peak:.1f}GB", flush=True)
        if limit == 25600:
            traceback.print_exc(limit=3)
        torch.cuda.empty_cache()
