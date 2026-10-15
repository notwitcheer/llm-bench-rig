"""Offline render of every frozen toolcall-v1 prompt (no GPU): think-on and think-off, official Qwen3.8
template, token counts with the official tokenizer, sha256 per prompt. Single-turn prompts in full;
multi-turn (S5) up to the first model turn (later turns grow with preserved thinking = model output).
Also re-renders with an alternative template (the one embedded in the unsloth Q6_K / UD-IQ3_XXS GGUFs)
and records whether the bytes differ.

usage: python render_prompts.py <manifest.json> <tokenizer dir> <alt template .jinja> <out dir>
"""
import gzip
import hashlib
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from transformers import AutoTokenizer  # noqa: E402

import bfcl_items as B  # noqa: E402
import qwen38_xml as X  # noqa: E402


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main(man_p, tok_dir, alt_tpl_p, out_dir):
    man = json.loads(Path(man_p).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(tok_dir)
    alt = AutoTokenizer.from_pretrained(tok_dir)
    alt.chat_template = Path(alt_tpl_p).read_text()
    rows = []
    with gzip.open(out / "prompts_toolcall-v1.jsonl.gz", "wt") as fz:
        for d in man["items"]:
            sets = [s for s in d["sets"] if s != "S5_phase2"]
            if not sets:
                continue
            e = B.entry(d["id"])
            tools = B.tools_for(e)
            msgs = B.messages_for(e)
            for think in (True, False):
                p = X.render_prompt(tok, msgs, tools, think)
                pa = X.render_prompt(alt, msgs, tools, think)
                n = len(tok.encode(p, add_special_tokens=False))
                r = {"id": d["id"], "sets": sets, "think": think, "sha256": hashlib.sha256(p.encode()).hexdigest(),
                     "n_tokens": n, "alt_template_identical": p == pa}
                rows.append(r)
                fz.write(json.dumps({**r, "prompt": p}) + "\n")
    (out / "prompt_index.json").write_text(json.dumps(rows, indent=0))
    summ = {}
    for think in (True, False):
        for s in ("S1", "S2", "S3", "S4", "S5"):
            xs = [r["n_tokens"] for r in rows if r["think"] == think and s in r["sets"]]
            if xs:
                summ[f"{s}_{'on' if think else 'off'}"] = {"n": len(xs), "p50": statistics.median(xs),
                                                           "p90": pct(xs, 0.9), "max": max(xs)}
    summ["alt_template_differs"] = sum(1 for r in rows if not r["alt_template_identical"])
    summ["alt_template_differs_ids"] = sorted({r["id"] for r in rows if not r["alt_template_identical"]})[:20]
    summ["max_prompt_tokens_single_turn"] = max(r["n_tokens"] for r in rows if "S5" not in r["sets"])
    summ["fits_ctx_24576_with_16384_budget"] = summ["max_prompt_tokens_single_turn"] + 16384 <= 24576
    (out / "prompt_stats.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
