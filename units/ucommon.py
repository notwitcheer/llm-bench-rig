"""shared pieces of the units experiment: request building, response parsing, the per-call classifier (exact rationals), unit humanising.
no third-party imports: the whole scoring path is stdlib."""
import hashlib
import json
import os
import re
from fractions import Fraction

HERE = os.path.dirname(os.path.abspath(__file__))
PARSER_VERSION = "units-parse-v1"
ROLES = ["canonical", "sub", "larger", "words"]
SEED = 20260725
CATS = ["ok", "magnitude_slip", "other_value", "no_call", "wrong_tool", "malformed"]


def sha(s):
    return hashlib.sha256(s.encode() if isinstance(s, str) else s).hexdigest()


def load_tasks(path=None):
    return json.load(open(path or os.path.join(HERE, "tasks.json")))


def load_models(path=None):
    return json.load(open(path or os.path.join(HERE, "models.json")))


def build_payload(spec, task, form, mcfg):
    """one single-turn tool-call request: the task's own tool only, tool_choice left at the server default (auto), greedy, seed fixed."""
    tool = spec["tools"][task["tool"]]
    payload = {
        "model": mcfg["alias"],
        "messages": [{"role": "system", "content": spec["system"]}, {"role": "user", "content": form["prompt_user"]}],
        "tools": [{"type": "function", "function": tool["schema"]}],
        "temperature": 0, "top_k": 1, "top_p": 1, "min_p": 0, "seed": SEED,
        "max_tokens": mcfg["max_tokens"], "stream": False, "cache_prompt": False,
        "chat_template_kwargs": mcfg["chat_template_kwargs"],
    }
    return payload


def parse_message(body):
    """pull the first tool call out of an OpenAI-style chat completion body. never raises."""
    out = {"n_calls": 0, "name": None, "args_raw": None, "content": "", "finish_reason": None, "reasoning_chars": 0, "error": None}
    try:
        ch = body["choices"][0]
        msg = ch.get("message") or {}
        out["finish_reason"] = ch.get("finish_reason")
        out["content"] = msg.get("content") or ""
        out["reasoning_chars"] = len(msg.get("reasoning_content") or "")
        tcs = msg.get("tool_calls") or []
        out["n_calls"] = len(tcs)
        if tcs:
            fn = tcs[0].get("function") or {}
            out["name"] = fn.get("name")
            a = fn.get("arguments")
            out["args_raw"] = a if isinstance(a, str) else json.dumps(a)
    except Exception as e:  # a body we cannot read is a no_call
        out["error"] = repr(e)
    return out


_NUM_STR = re.compile(r"^[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?$")


def parse_args(raw):
    """exact JSON parse: every number becomes a Fraction. returns (dict|None, err)."""
    try:
        obj = json.loads(raw, parse_float=Fraction, parse_int=Fraction)
    except Exception:
        return None, "bad_json"
    if not isinstance(obj, dict):
        return None, "args_not_object"
    return obj, None


def classify(task, form, parsed):
    """-> dict(cat, value (str|None), k, ratio, copied_stated, detail). categories are the pre-registered six."""
    expected = Fraction(task["expected"])
    r = {"cat": None, "value": None, "k": None, "ratio": None, "copied_stated": False, "detail": ""}
    if parsed["n_calls"] == 0 or parsed["name"] is None:
        r["cat"] = "no_call"
        r["detail"] = "finish=%s" % parsed["finish_reason"]
        return r
    if parsed["name"] != task["tool"]:
        r["cat"] = "wrong_tool"
        r["detail"] = "called %s" % parsed["name"]
        return r
    args, err = parse_args(parsed["args_raw"])
    if args is None:
        r["cat"] = "malformed"
        r["detail"] = err
        return r
    for key, want in task["must"].items():
        if args.get(key) != want:
            r["cat"] = "wrong_tool"
            r["detail"] = "%s=%r" % (key, args.get(key) if not isinstance(args.get(key), Fraction) else str(args.get(key)))
            return r
    if task["arg"] not in args:
        r["cat"] = "malformed"
        r["detail"] = "missing_arg"
        return r
    v = args[task["arg"]]
    if isinstance(v, bool) or v is None or isinstance(v, (list, dict)):
        r["cat"] = "malformed"
        r["detail"] = "not_a_number"
        return r
    if isinstance(v, str):
        s = v.strip()
        if _NUM_STR.match(s):
            v = Fraction(s)
            r["detail"] = "numeric_string"
        else:
            r["cat"] = "malformed"
            r["detail"] = "string_value:%s" % s[:30]
            return r
    r["value"] = str(v)
    r["ratio"] = str(v / expected) if expected != 0 else None
    r["copied_stated"] = form["role"] != "canonical" and v == Fraction(form["num"]) and v != expected
    if v == expected:
        r["cat"] = "ok"
        return r
    if v > 0:
        for k in (1, 2, 3, 4, 5, 6, -1, -2, -3, -4, -5, -6):
            if v * Fraction(10) ** k == expected:
                r["cat"] = "magnitude_slip"
                r["k"] = k
                return r
    r["cat"] = "other_value"
    return r


# ---- humanising a value for the line-1 sentence ----
_LADDERS = {
    "seconds": [("days", Fraction(86400)), ("hours", Fraction(3600)), ("minutes", Fraction(60)), ("seconds", Fraction(1)), ("milliseconds", Fraction(1, 1000))],
    "milliseconds": [("hours", Fraction(3600000)), ("minutes", Fraction(60000)), ("seconds", Fraction(1000)), ("milliseconds", Fraction(1)), ("microseconds", Fraction(1, 1000))],
    "metres": [("kilometres", Fraction(1000)), ("metres", Fraction(1)), ("centimetres", Fraction(1, 100)), ("millimetres", Fraction(1, 1000))],
    "grams": [("kilograms", Fraction(1000)), ("grams", Fraction(1)), ("milligrams", Fraction(1, 1000))],
}


def _num(f):
    f = Fraction(f)
    if f.denominator == 1:
        return f"{f.numerator:,}"
    s = ("%.4f" % float(f)).rstrip("0").rstrip(".")
    return s


def humanise(value, fam):
    """a value in the argument's own unit -> the most natural exact phrase (largest unit giving at most 2 decimals), e.g. 90000 seconds -> '25 hours'."""
    v = Fraction(value)
    if fam == "cents":
        if abs(v) < 100 or (v * 1) % 1 != 0:
            return "%s cent%s" % (_num(v), "" if v == 1 else "s")
        d = v / 100
        return "$" + (f"{d.numerator:,}" if d.denominator == 1 else f"{float(d):,.2f}")
    for name, f in _LADDERS[fam]:
        q = v / f
        if abs(q) >= 1 and (q * 100).denominator == 1:
            unit = name[:-1] if q == 1 else name
            return f"{_num(q)} {unit}"
    for name, f in _LADDERS[fam]:
        if abs(v) >= f:
            return f"{_num(v / f)} {name}"
    name, f = _LADDERS[fam][-1]
    return f"{_num(v / f)} {name}"
