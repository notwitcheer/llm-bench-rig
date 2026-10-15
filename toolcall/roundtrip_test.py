"""Parser round-trip test (design doc: "the parser is part of the measurement"), no GPU.

For every S1..S3 item of the frozen list (S2 included): take the ground truth, pick one canonical
acceptable value per parameter (first acceptable value; a leading "" = optional, omitted), render the
calls through the Qwen3.8 template's OWN assistant tool_calls branch, parse the rendered text back with
qwen38_xml.parse_reply (think-on path), and require
  (1) bucket == ok, (2) a lossless, type-strict match with the canonical calls,
  (3) BFCL's unchanged ast_checker passes the parsed calls.
Plus parser negative controls (each malformed shape must land in its bucket).

usage: python roundtrip_test.py <frozen manifest.json> <tokenizer dir> <out.json>; exit 0 only if all pass.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from transformers import AutoTokenizer  # noqa: E402

import bfcl_items as B  # noqa: E402
import qwen38_xml as X  # noqa: E402


canonical = B.canonical


def negative_controls():
    tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
        "s": {"type": "string"}, "n": {"type": "integer"}, "xs": {"type": "array", "items": {"type": "integer"}},
        "b": {"type": "boolean"}}, "required": ["s"]}}}]
    ok = "<tool_call>\n<function=f>\n<parameter=s>\nhi\n</parameter>\n<parameter=n>\n3\n</parameter>\n</function>\n</tool_call>"
    cases = [
        ("well-formed", "r\n</think>\n\n" + ok, "eos", "ok"),
        ("two calls", "\n</think>\n\n" + ok + "\n" + ok, "eos", "ok"),
        ("text before call allowed", "\n</think>\n\nI will call f.\n\n" + ok, "eos", "ok"),
        ("capped think", "thinking forever", "limit", "capped"),
        ("unterminated think, eos", "thinking", "eos", "malformed_structure"),
        ("unclosed parameter", "\n</think>\n\n<tool_call>\n<function=f>\n<parameter=s>\nhi\n</function>\n</tool_call>", "eos", "malformed_structure"),
        ("duplicate </parameter>", "\n</think>\n\n<tool_call>\n<function=f>\n<parameter=s>\nhi\n</parameter>\n</parameter>\n</function>\n</tool_call>", "eos", "malformed_structure"),
        ("missing wrapper", "\n</think>\n\n<function=f>\n<parameter=s>\nhi\n</parameter>\n</function>", "eos", "malformed_structure"),
        ("unclosed tool_call", "\n</think>\n\n<tool_call>\n<function=f>\n<parameter=s>\nhi\n</parameter>\n</function>\n", "eos", "malformed_structure"),
        ("stray text in block", "\n</think>\n\n<tool_call>\n<function=f>\nlet me see\n<parameter=s>\nhi\n</parameter>\n</function>\n</tool_call>", "eos", "malformed_structure"),
        ("bad int value", "\n</think>\n\n<tool_call>\n<function=f>\n<parameter=s>\nhi\n</parameter>\n<parameter=n>\nthree\n</parameter>\n</function>\n</tool_call>", "eos", "malformed_value"),
        ("bad array value", "\n</think>\n\n<tool_call>\n<function=f>\n<parameter=s>\nhi\n</parameter>\n<parameter=xs>\n[1, 2,\n</parameter>\n</function>\n</tool_call>", "eos", "malformed_value"),
        ("python bool (parser-2: accepted, flagged)", "\n</think>\n\n<tool_call>\n<function=f>\n<parameter=s>\nhi\n</parameter>\n<parameter=b>\nTrue\n</parameter>\n</function>\n</tool_call>", "eos", "ok"),
        ("bool written TRUE", "\n</think>\n\n<tool_call>\n<function=f>\n<parameter=s>\nhi\n</parameter>\n<parameter=b>\nTRUE\n</parameter>\n</function>\n</tool_call>", "eos", "malformed_value"),
        ("bool written yes", "\n</think>\n\n<tool_call>\n<function=f>\n<parameter=s>\nhi\n</parameter>\n<parameter=b>\nyes\n</parameter>\n</function>\n</tool_call>", "eos", "malformed_value"),
        ("json call leaked as text", '\n</think>\n\n{"name": "f", "arguments": {"s": "hi"}}', "eos", "leaked"),
        ("python call leaked as text", "\n</think>\n\nf(s='hi')", "eos", "leaked"),
        ("call only in reasoning", "I'll do <tool_call>\n<function=f>\n</think>\n\nSure, done.", "eos", "leaked"),
        ("plain answer", "\n</think>\n\nThe weather is nice.", "eos", "no_call"),
    ]
    res = []
    for name, text, stop, want in cases:
        p = X.parse_reply(text, tools, think=True, stop_type=stop)
        res.append({"case": name, "want": want, "got": p.bucket, "detail": p.detail, "pass": p.bucket == want})
    # parser-2: the python bool case decodes to a real boolean and carries the python_bool flag
    p = X.parse_reply(cases[12][1], tools, think=True, stop_type="eos")
    res.append({"case": "python bool -> True + flag", "pass": p.calls == [{"f": {"s": "hi", "b": True}}] and "python_bool" in p.flags})
    p = X.parse_reply(cases[0][1], tools, think=True, stop_type="eos")
    res.append({"case": "json path carries no python_bool flag", "pass": "python_bool" not in p.flags})
    return res


def main(manifest_path, tok_dir, out_path):
    man = json.loads(Path(manifest_path).read_text())
    tok = AutoTokenizer.from_pretrained(tok_dir)
    ids = [d["id"] for d in man["items"] if set(d["sets"]) & {"S1", "S2", "S3"}]
    fails, n = [], 0
    for i in ids:
        n += 1
        e = B.entry(i)
        tools = B.tools_for(e)
        canon = canonical(B.ground_truth(i), e["function"])
        text = X.render_assistant_tool_calls(tok, B.messages_for(e), tools, canon)
        p = X.parse_reply(text, tools, think=True, stop_type="eos")
        rec = {"id": i}
        if p.bucket != "ok":
            rec.update(stage="parse", bucket=p.bucket, detail=p.detail)
        elif not X.strict_equal(p.calls, canon):
            rec.update(stage="lossless", parsed=p.calls, canonical=canon)
        else:
            chk = B.check_single(i, e, p.calls, None)
            if not chk["valid"]:
                rec.update(stage="bfcl_checker", error_type=chk.get("error_type"), error=chk.get("error"), parsed=p.calls)
        if len(rec) > 1:
            rec["rendered"] = text[:600]
            fails.append(rec)
    neg = negative_controls()
    out = {"parser_version": X.PARSER_VERSION, "items_tested": n, "roundtrip_failures": len(fails),
           "failures": fails, "negative_controls": neg,
           "negative_controls_pass": all(c["pass"] for c in neg)}
    Path(out_path).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("parser_version", "items_tested", "roundtrip_failures", "negative_controls_pass")}))
    for f in fails[:15]:
        print("FAIL", json.dumps({k: v for k, v in f.items() if k != "rendered"}, default=str)[:400])
    for c in neg:
        if not c["pass"]:
            print("NEG FAIL", c)
    return 0 if (not fails and out["negative_controls_pass"]) else 1


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
