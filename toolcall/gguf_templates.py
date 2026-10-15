"""Read tokenizer.chat_template (+ a few tokenizer keys) from each row's GGUF header, CPU only, and diff
against the official Qwen3.8 chat_template.jinja. usage: python gguf_templates.py <official.jinja> <out.json> <gguf>..."""
import difflib
import hashlib
import json
import sys

from gguf import GGUFReader


def field_str(r, key):
    f = r.fields.get(key)
    if f is None:
        return None
    v = f.parts[f.data[0]]
    try:
        return bytes(v).decode("utf-8")
    except Exception:
        return v.tolist() if hasattr(v, "tolist") else str(v)


official = open(sys.argv[1]).read()
out = {"official_sha256": hashlib.sha256(official.encode()).hexdigest(), "rows": {}}
for p in sys.argv[3:]:
    r = GGUFReader(p)
    t = field_str(r, "tokenizer.chat_template")
    row = {"sha256": hashlib.sha256(t.encode()).hexdigest() if isinstance(t, str) else None,
           "identical_to_official": t == official,
           "identical_after_strip": isinstance(t, str) and t.strip() == official.strip(),
           "tokenizer.ggml.model": field_str(r, "tokenizer.ggml.model"),
           "tokenizer.ggml.pre": field_str(r, "tokenizer.ggml.pre"),
           "n_tokens": len(r.fields["tokenizer.ggml.tokens"].data) if "tokenizer.ggml.tokens" in r.fields else None,
           "general.name": field_str(r, "general.name")}
    for k in ("tokenizer.ggml.eos_token_id", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.add_bos_token"):
        f = r.fields.get(k)
        row[k] = f.parts[f.data[0]].tolist()[0] if f is not None else None
    if isinstance(t, str) and t != official:
        row["diff"] = "".join(list(difflib.unified_diff(official.splitlines(True), t.splitlines(True), "official", p, n=1))[:200])
    out["rows"][p] = row
    print(p.split("/")[-1], {k: v for k, v in row.items() if k != "diff"}, flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
