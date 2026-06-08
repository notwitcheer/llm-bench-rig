"""sentence-transformers encode path for all arms. Records throughput + peak VRAM."""
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from scripts.embed_bench.models import MODELS

MAX_SEQ_LEN = 512


def load(arm):
    cfg = MODELS[arm]
    model = SentenceTransformer(cfg["hf_id"], trust_remote_code=True, device="cuda")
    # The harness owns prompting via explicit MODELS prefixes. Disable any model-baked
    # default prompt (the official Qwen3-VL-Embedding repo sets one) so we don't double-prompt
    # and so documents never silently receive a query-style instruction.
    model.default_prompt_name = None
    # Cap sequence length at 512 for ALL arms: e5-small-v2 natively caps at 512, so this makes
    # the comparison apples-to-apples on SciFact's short abstracts (where 512 rarely truncates)
    # AND bounds per-batch activation memory so the encode fits alongside Donald.
    model.max_seq_length = MAX_SEQ_LEN
    return model, cfg


def encode_texts(model, texts, prefix, batch_size=64):
    """Encode with peak-VRAM + throughput capture. Returns (vecs, docs_per_sec, peak_gb)."""
    torch.cuda.reset_peak_memory_stats()
    prefixed = [prefix + t for t in texts]
    t0 = time.perf_counter()
    vecs = model.encode(
        prefixed, batch_size=batch_size, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=False,
    )
    dt = time.perf_counter() - t0
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    return np.asarray(vecs, dtype=np.float32), len(texts) / dt, peak_gb


def smoke(arm):
    model, cfg = load(arm)
    v, dps, peak = encode_texts(model, ["the capital of France is Paris"], cfg["query_prefix"], 1)
    norm = float(np.linalg.norm(v[0]))
    print(f"{arm}: dim={v.shape[1]} norm={norm:.3f} peak_vram={peak:.2f}GB ok={v.shape[1] > 0 and norm > 0.1}")


if __name__ == "__main__":
    import sys
    smoke(sys.argv[1])
