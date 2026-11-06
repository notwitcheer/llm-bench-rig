"""astra_common.py: shared constants + helpers for the Astra-Stop Q4 replay (item 90). Frozen by PREREG.md (sha256 in FROZEN.sha256)."""
import json, re, io, urllib.request

PROMPT = "Generate an SVG of a pelican riding a bicycle"
# Qwen's own thinking-mode sampler (model card, base revision 1d4bf0f). The Astra-Stop card does not state its sampler (UNVERIFIED), these are ours.
SAMP = dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repeat_penalty=1.0)
# reasoning_effort xhigh = the chat template default and the value the Astra-Stop card says it used; passed explicitly so the rendered system text is fixed.
KW = {"enable_thinking": True, "reasoning_effort": "xhigh"}
PEL_MAX_TOK = 60000          # per-slot context is 65536; a try that hits this is CAPPED (counted, never dropped)
MATH_MAX_TOK = 30000
SEEDS = list(range(1, 33))   # 32 seeds, the same list for both arms; legs take 1-16 then 17-32
MATH_N = 24
MATH_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def math_ids(n=MATH_N):
    import random
    return sorted(random.Random(20261105).sample(range(500), n))


def chat_body(prompt, seed, max_tokens, stream=True):
    b = dict(SAMP)
    b.update(model="astra", messages=[{"role": "user", "content": prompt}], seed=seed, max_tokens=max_tokens,
             stream=stream, chat_template_kwargs=KW)
    if stream:
        b["stream_options"] = {"include_usage": True}
    return b


def post_json(url, obj, timeout=60):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def tokenize_len(base, text):
    if not text:
        return 0
    return len(post_json(base + "/tokenize", {"content": text, "add_special": False, "parse_special": False}, 120)["tokens"])


def extract_svg(text):
    m = re.findall(r"<svg\b.*?</svg\s*>", text or "", re.S | re.I)
    return m[-1] if m else None


def svg_check(svg):
    """-> dict(closed, parses, renders, colours). complete = closed and parses and renders non-blank (finish_reason is checked by the caller)."""
    out = dict(closed=bool(svg), parses=False, renders=False, colours=0, elements=0)
    if not svg:
        return out
    from lxml import etree
    try:
        root = etree.fromstring(svg.encode("utf-8"), etree.XMLParser(recover=False, resolve_entities=False, no_network=True))
        out["parses"] = True
        out["elements"] = sum(1 for _ in root.iter())
    except Exception as e:
        out["parse_error"] = str(e)[:120]
        return out
    try:
        import resvg_py
        from PIL import Image
        s = svg
        if "xmlns=" not in s[:400]:
            s = s.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"', 1)
            out["xmlns_added"] = True
        png = resvg_py.svg_to_bytes(svg_string=s, width=384, background="#ffffff")
        im = Image.open(io.BytesIO(bytes(png))).convert("RGB").resize((96, 96))
        cols = im.getcolors(96 * 96) or []
        out["colours"] = len(cols)
        nonwhite = sum(n for n, c in cols if sum(c) < 740)
        out["nonwhite_frac"] = round(nonwhite / (96 * 96), 3)
        out["renders"] = len(cols) >= 4 and nonwhite / (96 * 96) >= 0.03
    except Exception as e:
        out["render_error"] = str(e)[:120]
    return out


def boxed(text):
    i = (text or "").rfind("\\boxed{")
    if i < 0:
        return None
    j = i + 7; d = 1; k = j
    while k < len(text) and d:
        d += (text[k] == "{") - (text[k] == "}"); k += 1
    return text[j:k - 1] if d == 0 else None


def norm_ans(a):
    if a is None:
        return None
    a = a.replace("\\left", "").replace("\\right", "").replace("\\!", "").replace("\\,", "").replace("dfrac", "frac").replace("tfrac", "frac")
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a)
    a = a.replace("^\\circ", "").replace("^{\\circ}", "").replace("$", "").replace(" ", "").rstrip(".")
    return a
