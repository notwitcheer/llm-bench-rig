#!/usr/bin/env python3
"""card-params.py: identify a model's maker-recommended run parameters BEFORE a voxel/bench run.

usage: card-params.py <hf_repo> [<hf_repo> ...] [--gguf /path/model.gguf]
  e.g. card-params.py meta-models/Muse-Glimmer-30B-GGUF --gguf ~/models/muse-glimmer/Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf

prints, per repo README (plus the base_model repo named in the card's front matter, one hop):
  every line naming sampling (temperature/top_p/top_k/min_p/presence/repetition penalty), output or reasoning budget
  (max tokens / output length / thinking budget), reasoning effort/strength knobs, chat-template kwargs and history notes
  (preserve reasoning_content); then the GGUF's general.sampling.* keys (what llama.cpp applies when no flag is passed)
  and the llama.cpp defaults that fill any gap. the human/agent writes the final values + source url + fetch date into
  voxelbench/model-params.md and passes them as SAMPLING_FLAGS / RB / EXTRA_FLAGS (day0-voxel.sh refuses without PARAMS_SRC).
written 2026-10-22 (mercury) after Muse ran on llama.cpp defaults because its GGUF carries no sampling keys.
"""
import re
import sys
import urllib.request

KEYS = re.compile(
    r"temperature|\btemp\b|--temp|top[_ -]?p|top[_ -]?k|min[_ -]?p|presence[_ ]penalty|repetition[_ ]penalty|repeat[_-]penalty|"
    r"max[_ ]?(new[_ ])?tokens|output (budget|length)|thinking budget|reasoning[_ -]budget|reasoning[_ ](effort|strength)|"
    r"enable_thinking|chat[_-]template[_-]kwargs|reasoning_content|sampling|best practices|recommended settings",
    re.I,
)
LLAMA_DEFAULTS = "llama.cpp defaults when neither flag nor GGUF key: temp 0.8, top-k 40, top-p 0.95, min-p 0.05, n_predict -1"


def fetch(repo):
    for br in ("main", "master"):
        url = f"https://huggingface.co/{repo}/raw/{br}/README.md"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "card-params"}), timeout=25) as r:
                return url, r.read().decode("utf-8", "replace")
        except Exception:
            continue
    return None, ""


def base_models(text):
    m = re.match(r"^---\n(.*?)\n---", text, re.S)
    if not m:
        return []
    fm = m.group(1)
    out = []
    mm = re.search(r"^base_model:\s*(.*?)(?=^\w|\Z)", fm, re.S | re.M)
    if mm:
        out = re.findall(r"([\w.-]+/[\w.-]+)", mm.group(1))
    return out[:2]


def scan(repo, seen):
    if repo in seen:
        return
    seen.add(repo)
    url, text = fetch(repo)
    print(f"\n== {repo}  ({url or 'README not reachable'})")
    if not text:
        return
    hits = 0
    in_code = False
    for i, line in enumerate(text.splitlines(), 1):
        if line.strip().startswith("```"):
            in_code = not in_code
        if KEYS.search(line) and len(line) < 400:
            print(f"  {i:4d}: {line.strip()[:240]}")
            hits += 1
    if not hits:
        print("  (no parameter lines found: check the card by hand, and the original model's card)")
    for b in base_models(text):
        if b != repo:
            scan(b, seen)


def gguf_keys(path):
    print(f"\n== GGUF {path}")
    try:
        try:
            from gguf import GGUFReader
        except ImportError:
            sys.path.insert(0, "/home/witcheer/llama.cpp-donald/gguf-py")
            from gguf import GGUFReader
        r = GGUFReader(path)
        n = 0
        for k, f in r.fields.items():
            if k.startswith("general.sampling"):
                try:
                    v = f.parts[f.data[0]].tolist()
                except Exception:
                    v = "?"
                print(f"  {k} = {v}")
                n += 1
        if not n:
            print("  NO general.sampling keys: without flags the run gets llama.cpp defaults, so SAMPLING_FLAGS are mandatory")
    except Exception as e:
        print(f"  could not read: {e}")
    print(f"  ({LLAMA_DEFAULTS})")


if __name__ == "__main__":
    args = sys.argv[1:]
    gg = None
    if "--gguf" in args:
        i = args.index("--gguf")
        gg = args[i + 1]
        args = args[:i] + args[i + 2:]
    if not args and not gg:
        print(__doc__)
        sys.exit(2)
    seen = set()
    for repo in args:
        scan(repo, seen)
    if gg:
        gguf_keys(gg)
