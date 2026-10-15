"""Raw-mode Qwen3.8 tool-call rendering + parsing for the toolcall-v1 slice (BFCL v4 subset).

Design: 99-personal/benchmark-rig/drafts/2026-10-15-toolcall-slice-design.md (vault), operator go 2026-10-15.

Raw mode = the prompt is rendered CLIENT-side with the official Qwen3.8 chat template
(Qwen/Qwen3.8-27B @ 1d4bf0f2, chat_template.jinja sha256 c3cf9e34...), sent as a plain string to
llama-server's /completion endpoint, and the reply is parsed HERE, not by llama.cpp's chat parser.
Both binaries (mainline, PrismML fork) therefore see byte-identical prompts and one parser grades
every row.

Qwen3.8 tool-call format (template, assistant `tool_calls` branch):

    <tool_call>
    <function=NAME>
    <parameter=KEY>
    VALUE
    </parameter>
    </function>
    </tool_call>

VALUE is the raw string for string arguments and `tojson` for everything else. So "malformed JSON
args" can only happen on array / object / integer / number / boolean parameters.

Buckets (one per reply to an item that expects a call), decided in this order:
  capped               think-on chain hit max_tokens before </think> (or think-off reply cut mid-call)
  malformed_structure  a call was attempted but no well-formed block (unclosed tags, duplicate
                       </parameter>, stray text inside the block, <function=...> without <tool_call>)
  malformed_value      structure fine, a non-string-typed parameter does not json.loads
  leaked               no call in the answer, but call-shaped text is (JSON {"name":..}, name(...)),
                       or a call appears only inside the reasoning
  no_call              plain text answer
  (well-formed)        handed to BFCL's unchanged ast_checker -> schema_invalid / wrong_call / correct
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from typing import Any

PARSER_VERSION = "toolcall-v1-parser-2"
# parser-2 (2026-10-15, after the step-1 probe, operator go): a boolean parameter written as the Python
# literal True / False (exact casing, surrounding whitespace ignored) is accepted as a boolean and flagged
# "python_bool" (format-adherence column). parser-1 counted it malformed_value. Nothing else changed.

TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"
FUNC_RE = re.compile(r"<function=([^>\n]*)>")
PARAM_OPEN_RE = re.compile(r"<parameter=([^>\n]*)>")
PARAM_CLOSE = "</parameter>"
FUNC_CLOSE = "</function>"

# schema types (OpenAI-converted, see bfcl convert_to_tool / GORILLA_TO_OPENAPI) that go through tojson
JSON_TYPES = {"array", "object", "integer", "number", "boolean"}


# --------------------------------------------------------------------------- rendering

def render_prompt(tokenizer, messages: list[dict], tools: list[dict] | None, think: bool) -> str:
    """Render with the official template via transformers (Jinja2 sandbox, HF tojson)."""
    kw: dict[str, Any] = dict(tokenize=False, add_generation_prompt=True, enable_thinking=think)
    if tools:
        kw["tools"] = tools
    return tokenizer.apply_chat_template(messages, **kw)


def render_assistant_tool_calls(tokenizer, messages: list[dict], tools: list[dict], calls: list[dict]) -> str:
    """Render ground-truth calls through the template's own assistant `tool_calls` branch and return
    the generated-side text exactly as a model would have to emit it after the think-on generation
    prompt `<|im_start|>assistant\\n<think>\\n` (i.e. starting with the reasoning, empty here)."""
    msgs = list(messages) + [
        {"role": "assistant", "content": "", "reasoning_content": "",
         "tool_calls": [{"type": "function", "function": {"name": n, "arguments": a}}
                        for c in calls for n, a in c.items()]},
    ]
    full = tokenizer.apply_chat_template(msgs, tools=tools, tokenize=False, add_generation_prompt=False,
                                         enable_thinking=True)
    marker = "<|im_start|>assistant\n<think>\n"
    i = full.rindex(marker) + len(marker)
    tail = full[i:]
    assert tail.endswith("<|im_end|>\n"), repr(tail[-40:])
    return tail[: -len("<|im_end|>\n")]


# --------------------------------------------------------------------------- schema helpers

def tool_param_types(tools: list[dict]) -> dict[str, dict[str, str]]:
    """{function_name: {param: openai_type}} from OpenAI-style tool dicts."""
    out: dict[str, dict[str, str]] = {}
    for t in tools:
        f = t.get("function", t)
        props = (f.get("parameters") or {}).get("properties") or {}
        out[f["name"]] = {k: (v.get("type") if isinstance(v, dict) else None) or "string" for k, v in props.items()}
    return out


# --------------------------------------------------------------------------- parsing

@dataclass
class Parsed:
    bucket: str                       # capped | malformed_structure | malformed_value | leaked | no_call | ok
    calls: list[dict] = field(default_factory=list)          # strict decode [{name: {k: v}}]
    calls_lenient: list[dict] | None = None                  # literal_eval fallback for values (sensitivity only)
    reasoning: str = ""
    answer: str = ""
    detail: str = ""
    flags: list[str] = field(default_factory=list)


def split_reasoning(generated: str, think: bool) -> tuple[str, str | None]:
    """Think-on: the prompt ends with '<think>\\n', so the generated text is reasoning + '</think>' +
    answer. Returns (reasoning, answer); answer is None when </think> never came."""
    if not think:
        return "", generated
    if "</think>" not in generated:
        return generated, None
    r, a = generated.split("</think>", 1)
    return r.strip(), a


def _strip_one_nl(v: str) -> str:
    if v.startswith("\n"):
        v = v[1:]
    if v.endswith("\n"):
        v = v[:-1]
    return v


def _convert(raw: str, typ: str | None, lenient: bool) -> tuple[bool, Any]:
    if typ is None or typ == "string":
        return True, raw
    try:
        return True, json.loads(raw)
    except Exception:
        pass
    if typ == "boolean" and raw.strip() in ("True", "False"):
        return True, raw.strip() == "True"
    if lenient:
        s = raw.strip()
        low = s.lower()
        if typ == "boolean" and low in ("true", "false"):
            return True, low == "true"
        try:
            return True, ast.literal_eval(s)
        except Exception:
            return False, raw
    return False, raw


def _parse_block(block: str, types: dict[str, dict[str, str]]):
    """block = text strictly between <tool_call> and </tool_call>.
    returns (status, name, raw_params | detail)"""
    body = block.strip()
    m = FUNC_RE.match(body)
    if not m:
        return "structure", None, "no <function=...> at block start"
    name = m.group(1).strip()
    if not body.endswith(FUNC_CLOSE):
        return "structure", name, "block does not end with </function>"
    inner = body[m.end(): -len(FUNC_CLOSE)]
    if FUNC_RE.search(inner) or FUNC_CLOSE in inner:
        return "structure", name, "more than one <function> in one <tool_call>"
    params: dict[str, str] = {}
    pos = 0
    while True:
        gap_m = re.compile(r"\s*").match(inner, pos)
        pos = gap_m.end()
        if pos >= len(inner):
            break
        pm = PARAM_OPEN_RE.match(inner, pos)
        if not pm:
            return "structure", name, f"stray text inside block: {inner[pos:pos+60]!r}"
        key = pm.group(1).strip()
        close = inner.find(PARAM_CLOSE, pm.end())
        if close < 0:
            return "structure", name, f"unclosed <parameter={key}>"
        raw = inner[pm.end(): close]
        if PARAM_OPEN_RE.search(raw):
            return "structure", name, f"<parameter={key}> not closed before next <parameter>"
        if key in params:
            return "structure", name, f"duplicate parameter {key}"
        params[key] = _strip_one_nl(raw)
        pos = close + len(PARAM_CLOSE)
    return "ok", name, params


def _leak_signals(text: str, fn_names: list[str]) -> list[str]:
    sig = []
    if re.search(r'\{\s*"name"\s*:\s*"', text) and re.search(r'"(arguments|parameters)"\s*:', text):
        sig.append("json_call_in_text")
    for n in fn_names:
        if re.search(r"(?<![\w.])" + re.escape(n) + r"\s*\(", text):
            sig.append(f"pycall_in_text:{n}")
            break
    return sig


def parse_reply(generated: str, tools: list[dict], think: bool, stop_type: str | None = None) -> Parsed:
    types = tool_param_types(tools)
    reasoning, answer = split_reasoning(generated, think)
    if answer is None:
        return Parsed("capped" if stop_type == "limit" else "malformed_structure", reasoning=reasoning,
                      detail="no </think> in generated text" + ("" if stop_type == "limit" else f" (stop_type={stop_type})"))
    p = Parsed("no_call", reasoning=reasoning, answer=answer)
    if "<think>" in answer or "</think>" in answer:
        p.flags.append("think_tag_in_answer")
    has_open = TOOL_OPEN in answer
    has_func = FUNC_RE.search(answer) is not None
    if not has_open and not has_func and TOOL_CLOSE not in answer:
        fn_names = list(types.keys())
        sig = _leak_signals(answer, fn_names)
        if think and (TOOL_OPEN in reasoning or FUNC_RE.search(reasoning)):
            sig.append("call_only_in_reasoning")
        if sig:
            p.bucket, p.detail, p.flags = "leaked", ",".join(sig), p.flags + sig
        return p
    # walk the blocks
    blocks, outside, pos = [], [], 0
    while True:
        i = answer.find(TOOL_OPEN, pos)
        if i < 0:
            outside.append(answer[pos:])
            break
        outside.append(answer[pos:i])
        j = answer.find(TOOL_CLOSE, i + len(TOOL_OPEN))
        if j < 0:
            p.bucket = "capped" if (stop_type == "limit" and not think) else "malformed_structure"
            p.detail = "unclosed <tool_call>"
            return p
        inner = answer[i + len(TOOL_OPEN): j]
        if TOOL_OPEN in inner:
            p.bucket, p.detail = "malformed_structure", "nested/unclosed <tool_call>"
            return p
        blocks.append(inner)
        pos = j + len(TOOL_CLOSE)
    out_text = "".join(outside)
    if TOOL_CLOSE in out_text:
        p.bucket, p.detail = "malformed_structure", "stray </tool_call>"
        return p
    if FUNC_RE.search(out_text) or PARAM_OPEN_RE.search(out_text):
        p.bucket, p.detail = "malformed_structure", "<function=...>/<parameter=...> outside <tool_call> (missing wrapper)"
        return p
    if outside[-1].strip():
        p.flags.append("text_after_calls")
    strict_calls, lenient_calls, value_errs = [], [], []
    for b in blocks:
        st, name, res = _parse_block(b, types)
        if st != "ok":
            p.bucket, p.detail = "malformed_structure", res
            return p
        ftypes = types.get(name, {})
        sargs, largs = {}, {}
        for k, raw in res.items():
            t = ftypes.get(k)
            if t == "boolean" and raw.strip() in ("True", "False") and "python_bool" not in p.flags:
                p.flags.append("python_bool")
            ok, v = _convert(raw, t, lenient=False)
            if not ok:
                value_errs.append(f"{name}.{k}:{t}={raw[:40]!r}")
            sargs[k] = v
            okl, vl = _convert(raw, t, lenient=True)
            largs[k] = vl
        strict_calls.append({name: sargs})
        lenient_calls.append({name: largs})
    p.calls_lenient = lenient_calls
    if value_errs:
        p.bucket, p.detail = "malformed_value", "; ".join(value_errs)
        return p
    p.bucket, p.calls = "ok", strict_calls
    return p


def strict_equal(a: Any, b: Any) -> bool:
    """Equality that also distinguishes bool/int/float (1 vs 1.0 vs True)."""
    if type(a) is not type(b):
        return False
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(strict_equal(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return len(a) == len(b) and all(strict_equal(x, y) for x, y in zip(a, b))
    return a == b
