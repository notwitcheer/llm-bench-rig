#!/usr/bin/env python3
"""rc_common.py: shared code of the replay-cache harness (CPU-ARC near-miss reproduction on GSM-Symbolic). stdlib only.
Frozen: the prompts, the parser, the scorer and the statistics live here and are sha256-listed in FROZEN.sha256."""
import json, math, os, random, re, subprocess, sys, threading, time, urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(HERE)
DATA = os.environ.get("RC_DATA", os.path.join(BASE, "data"))
SEED = 20261109
N_TEMPLATES = 20
TOL = 0.005          # a number is right when |x - golden| < TOL

# ---------------------------------------------------------------------------------------------------------------- prompts (frozen)
SRC_PROMPT = ("Solve this problem. Show your reasoning, then give the final numerical answer on the last line in the form "
              "`#### <number>` (digits only, no units, no commas).\n\n{question}")

BUILD_PROMPT = """You are compiling a worked solution into a reusable program.

Below is a maths word problem and the transcript of how it was solved from scratch. The transcript's final answer is {answer}.
Write ONE Python function that solves this whole family of problems when the names and numbers in the question change.

Reply with exactly two fenced blocks.
Block 1, fenced as ```json, an object with three keys:
  "description": one line saying what family of question this function solves (no numbers from this instance),
  "schema": a list of objects {{"name": snake_case_name, "description": what to read off the question, "example": the value in THIS question}},
  "params": an object mapping every schema name to its value in THIS question.
Block 2, fenced as ```python, a function `def solve(params: dict) -> float:` that reads every value from `params`, does all the arithmetic and returns a number. Standard library only (math, fractions).

Rules: every param is a plain number that can be read straight off the question (write a fraction as a numerator and a denominator, a percent as the number before the % sign, a number word such as "three" as 3, "twice" as 2); any conversion the question does not state belongs inside solve(). With this question's params solve() must return {answer}. Do not hard-code {answer} or any value from this question inside solve(); changing a param must change the result.
The numbers of this one question can hide the general rule (a fraction of 1/2 makes "the part that left" equal "the part that stayed", two equal numbers hide which one a quantity refers to). Work out from the wording of the question what each quantity means, so that solve() stays right for other fractions, other numbers and other orderings of the same story.

QUESTION:
{question}

TRANSCRIPT:
{transcript}
"""

BUILD_FIX = ("Your function returned {got} for this question's params, but the transcript's final answer is {answer}. "
             "Problem: {why}. Fix the function and/or the params, and reply with both fenced blocks again.")

EXTRACT_PROMPT = """You fill in the parameters of a saved solver. The solver handles this kind of question: {description}

Worked example.
Question: {build_question}
Parameters: {example_params}

Parameter meanings:
{keys}

Now read the new question and reply with a JSON object holding exactly those keys, each a plain number (a number word such as "three" is 3; a fraction is its numerator and denominator when the keys say so).

Question: {question}
Parameters:"""

C_COT_PROMPT = ("Solve this problem. Think step by step, briefly, then give the final numerical answer on the last line in the form "
                "`#### <number>` (digits only, no units, no commas).\n\n{question}")
C_DIRECT_PROMPT = "Answer with only the final number (digits only, no units, no commas, no words).\n\n{question}"

# ---------------------------------------------------------------------------------------------------------------- data + selection
def load_rows(split):
    p = os.path.join(DATA, split, "test.jsonl")
    return [json.loads(l) for l in open(p)]


def index_rows(split):
    return {(r["original_id"], r["instance"]): r for r in load_rows(split)}


def file_sha(path):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def make_selection(n_templates=N_TEMPLATES, n_held=10, n_near=10):
    main, p1 = index_rows("main"), index_rows("p1")
    common = sorted(set(o for o, _ in main) & set(o for o, _ in p1))
    rng = random.Random(SEED)
    tids = rng.sample(common, 20)[:n_templates]            # the first 20-draw is the pre-registered template pick; a mock may take a prefix
    out = {"seed": SEED, "n_held": n_held, "n_near": n_near, "source": "apple/GSM-Symbolic main + p1 (CC-BY-NC-ND 4.0, eval only; ids only here)",
           "data_sha256": {"main/test.jsonl": file_sha(os.path.join(DATA, "main", "test.jsonl")), "p1/test.jsonl": file_sha(os.path.join(DATA, "p1", "test.jsonl"))},
           "templates": []}
    for t in tids:
        held = sorted(rng.sample(range(1, 50), n_held))
        near = sorted(rng.sample(range(0, 50), n_near))
        assert (t, 0) in main and all((t, i) in main for i in held) and all((t, i) in p1 for i in near)
        out["templates"].append({"tid": t, "build": ["main", t, 0], "held": [["main", t, i] for i in held], "near": [["p1", t, i] for i in near]})
    return out


def items_of(sel, kind, limit_templates=None):
    ts = sel["templates"][:limit_templates] if limit_templates else sel["templates"]
    key = {"held": "held", "near": "near"}[kind]
    n = max(len(t[key]) for t in ts)
    order = []
    for i in range(n):                                      # interleaved: a cut run leaves every template covered
        for t in ts:
            if i < len(t[key]):
                order.append((kind, t["tid"], t[key][i][2]))
    return order


def limited_items(sel, lt=None, n=0):
    """smoke/mock only: the first n items = ceil(n/2) held-out + floor(n/2) near-miss, so a limited run still crosses the p1 path. n=0 = all."""
    h, nm = items_of(sel, "held", lt), items_of(sel, "near", lt)
    if not n:
        return h + nm
    return h[: (n + 1) // 2] + nm[: n // 2]


def item_key(kind, tid, inst):
    return "%s-%d-%02d" % (kind, tid, inst)


def row_for(kind, tid, inst, cache={}):
    split = "p1" if kind == "near" else "main"
    if split not in cache:
        cache[split] = index_rows(split)
    return cache[split][(tid, inst)]


# ---------------------------------------------------------------------------------------------------------------- numbers
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


def to_num(s):
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None


def golden_of(row):
    return to_num(row["answer"].split("####")[-1].strip().replace("$", ""))


def parse_final(text):
    """the number after the last '####'; else the last number in the text. returns (value, how)."""
    if not text:
        return None, "empty"
    if "####" in text:
        m = _NUM.search(text.rsplit("####", 1)[1])
        if m:
            return to_num(m.group(0)), "marker"
    ms = _NUM.findall(text)
    if ms:
        return to_num(ms[-1]), "last_number"
    return None, "none"


def same(a, b):
    return a is not None and b is not None and abs(a - b) < TOL


# ---------------------------------------------------------------------------------------------------------------- statistics
def wilson(k, n, z=1.959964):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    """two-sided exact McNemar on the discordant counts (b = only first right, c = only second right)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def pct(x, d=0):
    return ("%." + str(d) + "f") % (100 * x)


# ---------------------------------------------------------------------------------------------------------------- http
def post_json(url, obj, timeout=600):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(), headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def chat(base, messages, max_tokens, model="x", timeout=3600, **extra):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": False}
    body.update(extra)
    t0 = time.time()
    r = post_json(base + "/v1/chat/completions", body, timeout)
    r["_wall_s"] = time.time() - t0
    return r


def msg_of(r):
    m = r["choices"][0]["message"]
    return (m.get("content") or ""), (m.get("reasoning_content") or "")


def atomic_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp%d" % os.getpid()
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)


def solve_one(base, question, seed, max_tokens, model="qwen38-27b", timeout=3600):
    """arm A / the builder's source: one from-scratch completion with the house model (sampling = the server's defaults, set at launch)."""
    r = chat(base, [{"role": "user", "content": SRC_PROMPT.format(question=question)}], max_tokens, model=model, seed=seed, timeout=timeout,
             chat_template_kwargs={"enable_thinking": True})
    content, reasoning = msg_of(r)
    val, how = parse_final(content)
    u, tm = r.get("usage", {}), r.get("timings", {})
    return {"answer": val, "how": how, "content": content, "reasoning": reasoning, "finish": r["choices"][0].get("finish_reason"),
            "completion_tokens": u.get("completion_tokens"), "prompt_tokens": u.get("prompt_tokens"), "wall_s": round(r["_wall_s"], 2),
            "decode_tok_s": tm.get("predicted_per_second"), "seed": seed}


# ---------------------------------------------------------------------------------------------------------------- entry code: static filter + isolated runner
STATIC_BAD = re.compile(r"\b(import\s+(os|sys|subprocess|socket|shutil|pathlib|requests|urllib|ctypes|builtins)|__import__|open\s*\(|eval\s*\(|exec\s*\(|compile\s*\(|globals\s*\(|getattr\s*\(|input\s*\()")

_RUNNER = r'''
import sys, json, resource, math
resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30)); resource.setrlimit(resource.RLIMIT_CPU, (6, 6))
src = open(sys.argv[1]).read(); params = json.loads(sys.argv[2])
g = {"__name__": "entry"}
exec(compile(src, "entry", "exec"), g)
v = g["solve"](params)
if isinstance(v, bool): raise TypeError("bool result")
v = float(v)
if not math.isfinite(v): raise ValueError("non-finite result")
print(json.dumps({"v": v}))
'''


def run_solve(code_path, params, timeout=10):
    """run an entry's solve(params) in an isolated child. returns (ok, value|None, why)."""
    code_path = os.path.abspath(code_path)
    try:
        p = subprocess.run([sys.executable, "-I", "-c", _RUNNER, code_path, json.dumps(params)], capture_output=True, text=True, timeout=timeout, cwd="/tmp")
    except subprocess.TimeoutExpired:
        return False, None, "timeout"
    if p.returncode != 0:
        last = (p.stderr.strip().splitlines() or ["error"])[-1]
        return False, None, "crash: " + last[:160]
    try:
        return True, json.loads(p.stdout.strip().splitlines()[-1])["v"], ""
    except Exception:
        return False, None, "non-number output"


def parse_builder(text):
    """returns (dict|None, why). needs a ```json block (description, schema, params) and a ```python block with solve()."""
    jb = re.findall(r"```json\s*(.*?)```", text, re.S)
    pb = re.findall(r"```python\s*(.*?)```", text, re.S)
    if not jb or not pb:
        return None, "reply must contain one ```json block and one ```python block"
    try:
        j = json.loads(jb[-1])
    except Exception as e:
        return None, "json block does not parse (%s)" % str(e)[:80]
    if not isinstance(j, dict) or not isinstance(j.get("description"), str) or not isinstance(j.get("schema"), list) or not isinstance(j.get("params"), dict):
        return None, "json block needs description (str), schema (list), params (object)"
    names = []
    for s in j["schema"]:
        if not isinstance(s, dict) or not isinstance(s.get("name"), str) or "description" not in s or "example" not in s:
            return None, "every schema item needs name, description, example"
        names.append(s["name"])
    if not names or len(set(names)) != len(names):
        return None, "schema names must be non-empty and unique"
    for n in names:
        v = j["params"].get(n)
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None, "param %r is missing or not a number" % n
    code = pb[-1].strip() + "\n"
    if "def solve" not in code:
        return None, "python block has no def solve"
    if STATIC_BAD.search(code):
        return None, "code uses a forbidden construct (imports other than math/fractions, open, eval, exec, getattr)"
    return {"description": j["description"].strip(), "schema": j["schema"], "params": {n: j["params"][n] for n in names}, "code": code}, ""


def validate_entry(parsed, work_dir, want, tag):
    """run solve on the build params: must reproduce `want` (the SOURCE's answer, never the golden) and must not be constant in its params."""
    os.makedirs(work_dir, exist_ok=True)
    cp = os.path.join(work_dir, "cand-%s.py" % tag)
    open(cp, "w").write(parsed["code"])
    ok, v, why = run_solve(cp, parsed["params"])
    if not ok:
        return False, None, why
    if not same(v, want):
        return False, v, "result differs from the transcript's answer"
    moved = False
    for n, x in parsed["params"].items():
        p2 = dict(parsed["params"]); p2[n] = x * 2 + 1
        ok2, v2, _ = run_solve(cp, p2)
        if ok2 and v2 is not None and abs(v2 - v) > 1e-9:
            moved = True
            break
    if not moved:
        return False, v, "result does not change when a param changes (hard-coded)"
    return True, v, ""


def transcript_of(rec, cap=16000):
    t = (rec.get("reasoning") or "").strip()
    if len(t) > cap:
        t = t[: cap // 2] + "\n[... middle of the reasoning elided ...]\n" + t[-cap // 2:]
    return (t + "\n\nFINAL WORKED ANSWER:\n" + (rec.get("content") or "").strip()).strip()


def entry_text(ent):
    return ent["description"] + "\n" + ent["build_question"]


def log(*a):
    print("[rc] %s" % time.strftime("%H:%M:%S"), *a, flush=True)
