"""toolcall-v1 raw-mode driver: probe (and later, full legs) against a running llama-server.

Sends the client-rendered official-template prompt to llama-server's raw /completion endpoint
(temperature 0 = greedy, one request at a time), parses with qwen38_xml, grades with BFCL's unchanged
checkers (ast_checker single-turn; execute + multi_turn_checker multi-turn, BFCL loop semantics:
a step that decodes to no call ends the turn, more than MAXIMUM_STEP_LIMIT=20 steps force-quits
the entry). Transport errors are retried (gorilla #1339) and logged, never scored.

usage: python driver.py --server http://127.0.0.1:8097 --label <row> --ids <probe_ids.json>
         --manifest <frozen manifest> --tokenizer <dir> --prompts <prompts .jsonl.gz> --out <dir>
         [--think on|off] [--max-tokens 16384] [--ctx N] [--budget-s S] [--tokenize-all]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from transformers import AutoTokenizer  # noqa: E402

import bfcl_items as B  # noqa: E402
import qwen38_xml as X  # noqa: E402
from bfcl_eval.constants.default_prompts import MAXIMUM_STEP_LIMIT  # noqa: E402
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import multi_turn_checker  # noqa: E402
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call  # noqa: E402
from bfcl_eval.model_handler.utils import convert_to_function_call  # noqa: E402
from bfcl_eval.utils import load_ground_truth_entry  # noqa: E402

TRANSPORT_LOG = []


def post(server, path, body, timeout):
    data = json.dumps(body).encode()
    for attempt in range(4):
        try:
            req = urllib.request.Request(server + path, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except (urllib.error.URLError, ConnectionError, TimeoutError) as ex:
            TRANSPORT_LOG.append({"t": time.time(), "path": path, "attempt": attempt, "error": repr(ex)[:300]})
            if isinstance(ex, TimeoutError) or "timed out" in repr(ex):
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"transport failed 4x on {path}")


def get(server, path):
    with urllib.request.urlopen(server + path, timeout=30) as r:
        return json.loads(r.read())


def complete(server, prompt, n_predict, timeout, cache_prompt=False):
    """single-turn: cache_prompt False (every item prefilled from scratch); multi-turn steps: True
    (the next step's prompt extends the previous one)."""
    t0 = time.time()
    r = post(server, "/completion", {"prompt": prompt, "n_predict": n_predict, "temperature": 0.0, "seed": 0,
                                     "cache_prompt": cache_prompt, "stream": False}, timeout)
    r["_wall_s"] = time.time() - t0
    return r


def comp_meta(r):
    t = r.get("timings") or {}
    return {"stop_type": r.get("stop_type"), "completion_tokens": r.get("tokens_predicted"),
            "prompt_tokens": r.get("tokens_evaluated"), "truncated": r.get("truncated"),
            "wall_s": round(r["_wall_s"], 2), "prompt_ms": t.get("prompt_ms"), "prompt_n": t.get("prompt_n"),
            "predicted_ms": t.get("predicted_ms"), "decode_tok_s": t.get("predicted_per_second"),
            "cache_n": t.get("cache_n")}


def grade_single(i, e, tools, p: X.Parsed, is_irrel):
    """returns (outcome, bfcl_check, lenient_outcome)"""
    if is_irrel:
        if p.bucket == "capped":
            return "capped", {"valid": False, "error_type": "capped"}, "capped"
        decoded = p.calls if p.bucket == "ok" else None
        chk = B.check_single(i, e, decoded, p.detail)
        return ("correct" if chk["valid"] else "called_when_should_not"), chk, None
    if p.bucket != "ok":
        lo = p.bucket
        if p.bucket == "malformed_value" and p.calls_lenient:
            lc = B.check_single(i, e, p.calls_lenient, None)
            lo = B.outcome_after_check(lc, p.calls_lenient, tools)
        return p.bucket, None, lo
    chk = B.check_single(i, e, p.calls, None)
    oc = B.outcome_after_check(chk, p.calls, tools)
    return oc, chk, oc


def run_multi_turn(server, tok, i, think, max_tokens, ctx, deadline, label):
    e = B.entry(i)
    tools = B.tools_for(e)
    name_map = {f["name"].replace(".", "_"): f["name"] for f in e["function"]}
    gt = {g["id"]: g["ground_truth"] for g in load_ground_truth_entry("multi_turn_base")}[i]
    model_name = B.CHECKER_MODEL.replace("-", "_") + "_" + label.replace("-", "_")
    execute_multi_turn_func_call([], e.get("initial_config", {}), e["involved_classes"], model_name, i,
                                 long_context=False, is_evaL_run=False)
    msgs, decoded_all, steps_log = [], [], []
    force_quit, budget_abort, ctx_limited = False, False, False
    for t_idx, turn_msgs in enumerate(e["question"]):
        msgs.extend(turn_msgs)
        turn_decoded, count = [], 0
        while True:
            prompt = X.render_prompt(tok, msgs, tools, think)
            n_prompt = len(tok.encode(prompt, add_special_tokens=False))
            n_pred = min(max_tokens, ctx - n_prompt - 2)
            if n_pred < max_tokens:
                ctx_limited = True
            left = deadline - time.time()
            if left < 20 or n_pred < 64:
                budget_abort = left < 20
                force_quit = force_quit or n_pred < 64
                break
            try:
                r = complete(server, prompt, n_pred, timeout=left, cache_prompt=True)
            except Exception as ex:  # noqa: BLE001
                budget_abort = True
                steps_log.append({"turn": t_idx, "step": count, "error": repr(ex)[:200]})
                break
            gen = r.get("content", "")
            p = X.parse_reply(gen, tools, think, r.get("stop_type"))
            meta = comp_meta(r)
            steps_log.append({"turn": t_idx, "step": count, "parser": X.PARSER_VERSION, "n_prompt_hf": n_prompt, "n_predict": n_pred, **meta,
                              "bucket": p.bucket, "detail": p.detail, "flags": p.flags, "n_calls": len(p.calls),
                              "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "generated": gen})
            if p.bucket != "ok" or not p.calls:
                msgs.append({"role": "assistant", "content": (p.answer or "").strip(), "reasoning_content": p.reasoning})
                break
            calls = [{name_map.get(n, n): a} for c in p.calls for n, a in c.items()]
            strs = convert_to_function_call(calls)
            turn_decoded.append(strs)
            msgs.append({"role": "assistant", "content": "", "reasoning_content": p.reasoning,
                         "tool_calls": [{"type": "function", "function": {"name": n, "arguments": a}}
                                        for c in p.calls for n, a in c.items()]})
            results, _ = execute_multi_turn_func_call(strs, e.get("initial_config", {}), e["involved_classes"],
                                                      model_name, i, long_context=False, is_evaL_run=False)
            for res in results:
                msgs.append({"role": "tool", "content": res})
            count += 1
            if count > MAXIMUM_STEP_LIMIT:
                force_quit = True
                break
        decoded_all.append(turn_decoded)
        if force_quit or budget_abort:
            break
    rec = {"id": i, "turns_gt": len(gt), "turns_done": len(decoded_all), "steps": len(steps_log),
           "force_quit": force_quit, "budget_abort": budget_abort, "ctx_limited": ctx_limited,
           "max_prompt_tokens": max([s.get("n_prompt_hf", 0) for s in steps_log] or [0]),
           "max_prompt_plus_gen": max([s.get("n_prompt_hf", 0) + (s.get("completion_tokens") or 0) for s in steps_log] or [0]),
           "wall_s": round(sum(s.get("wall_s", 0) for s in steps_log), 1),
           "completion_tokens": sum((s.get("completion_tokens") or 0) for s in steps_log),
           "step_buckets": {}}
    for s in steps_log:
        if "bucket" in s:
            rec["step_buckets"][s["bucket"]] = rec["step_buckets"].get(s["bucket"], 0) + 1
    if budget_abort:
        rec["outcome"] = "budget_abort (unscored)"
    elif len(decoded_all) != len(gt):
        rec["outcome"] = "force_terminated"
    else:
        try:
            chk = multi_turn_checker(decoded_all, gt, e, "multi_turn_base", B.CHECKER_MODEL)
            rec["outcome"] = "correct" if chk.get("valid") else "wrong"
            rec["check"] = {k: v for k, v in chk.items() if k in ("valid", "error_type", "error_message")}
        except Exception as ex:  # noqa: BLE001
            rec["outcome"] = "checker_error"
            rec["check_error"] = repr(ex)[:300]
    rec["decoded"] = decoded_all
    rec["steps_log"] = steps_log
    return rec


def main():
    ap = argparse.ArgumentParser()
    for a in ("--server", "--label", "--ids", "--manifest", "--tokenizer", "--prompts", "--out", "--server-bin"):
        ap.add_argument(a, required=a not in ("--server-bin", "--ids"), default="")
    ap.add_argument("--sets", default="", help="build the id list from the frozen manifest, e.g. S1,S2,S3,S4 or S5 (phase 1)")
    ap.add_argument("--think", default="on")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--ctx", type=int, default=24576)
    ap.add_argument("--budget-s", type=int, default=900)
    ap.add_argument("--tokenize-all", action="store_true")
    a = ap.parse_args()
    t_start = time.time()
    deadline = t_start + a.budget_s
    think = a.think == "on"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    man = {d["id"]: d for d in json.loads(Path(a.manifest).read_text())["items"]}
    if a.ids:
        probe = json.loads(Path(a.ids).read_text())
    else:
        want = [x for x in a.sets.split(",") if x]
        assert want, "give --ids or --sets"
        items_m = json.loads(Path(a.manifest).read_text())["items"]
        probe = {s: [d["id"] for d in items_m if s in d["sets"]] for s in want}
    prompts = {}
    with gzip.open(a.prompts, "rt") as f:
        for line in f:
            r = json.loads(line)
            prompts[(r["id"], r["think"])] = r

    props = get(a.server, "/props")
    (out / "props.json").write_text(json.dumps(props, indent=1))
    tpl = props.get("chat_template", "")
    summary = {"label": a.label, "server_bin": a.server_bin, "think": a.think, "max_tokens": a.max_tokens, "ctx": a.ctx,
               "parser_version": X.PARSER_VERSION, "regime": f"raw /completion, official Qwen3.8 template client-side, think-{a.think}, max_tokens {a.max_tokens}, ctx {a.ctx}, greedy (temperature 0), preserve_thinking default (undefined = keep)",
               "props": {k: props.get(k) for k in ("total_slots", "build_info", "model_path")},
               "n_ctx": (props.get("default_generation_settings") or {}).get("n_ctx"),
               "embedded_template_sha256": hashlib.sha256(tpl.encode()).hexdigest() if tpl else None}

    # tokenizer parity: server /tokenize vs official HF tokenizer, every frozen prompt (or probe prompts only); once per leg dir
    par_p = out / "tokenize_parity.json"
    if par_p.exists():
        par = json.loads(par_p.read_text())
        summary["tokenize_parity"] = {"prompts": len(par), "mismatches_vs_hf": sum(1 for x in par if not x["identical"]), "seconds": None}
    else:
        keys = [k for k in prompts if a.tokenize_all or k[0] in set(sum(probe.values(), []))]
        par, mism = [], 0
        t0 = time.time()
        for k in keys:
            pr = prompts[k]["prompt"]
            ids_s = post(a.server, "/tokenize", {"content": pr, "add_special": False, "parse_special": True}, 60)["tokens"]
            ids_h = tok.encode(pr, add_special_tokens=False)
            same = ids_s == ids_h
            mism += not same
            par.append({"id": k[0], "think": k[1], "n_server": len(ids_s), "n_hf": len(ids_h), "identical": same,
                        "server_ids_sha256": hashlib.sha256(json.dumps(ids_s).encode()).hexdigest()})
        par_p.write_text(json.dumps(par))
        summary["tokenize_parity"] = {"prompts": len(par), "mismatches_vs_hf": mism, "seconds": round(time.time() - t0, 1)}
    print(json.dumps({"tokenize_parity": summary["tokenize_parity"]}), flush=True)

    def load_done(path):
        done = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    r = json.loads(line)
                    done[r["id"]] = r
        return done

    # single-turn: each id once (an item can sit in two sets), S1 then S2 first so the first-post scope lands first;
    # resumable: ids already in items.jsonl are skipped, a budget-aborted item is not written and reruns next time
    order, sets_of = [], {}
    for s in ("S1", "S2", "S3", "S4"):
        for i in probe.get(s, []):
            if i not in sets_of:
                order.append(i)
                sets_of[i] = []
            sets_of[i].append(s)
    st_p = out / "items.jsonl"
    done = load_done(st_p)
    aborted = 0
    with open(st_p, "a") as fo:
        for i in order:
            if i in done:
                continue
            left = deadline - time.time()
            if left < 30:
                aborted += 1
                continue
            e = B.entry(i)
            tools = B.tools_for(e)
            pr = prompts[(i, think)]["prompt"]
            assert pr == X.render_prompt(tok, B.messages_for(e), tools, think), f"prompt drift on {i}"
            try:
                r = complete(a.server, pr, a.max_tokens, timeout=left)
            except Exception as ex:  # noqa: BLE001
                aborted += 1
                TRANSPORT_LOG.append({"t": time.time(), "id": i, "error": repr(ex)[:300]})
                print(f"[{a.label}] {i}: request aborted {repr(ex)[:200]}", flush=True)
                continue
            gen = r.get("content", "")
            p = X.parse_reply(gen, tools, think, r.get("stop_type"))
            oc, chk, lo = grade_single(i, e, tools, p, "S4" in sets_of[i])
            rec = {"id": i, "set": sets_of[i][0], "sets": sets_of[i],
                   "tags": {k: man[i].get(k) for k in ("n_calls", "nested", "json_arg", "gt_nested")},
                   "parser": X.PARSER_VERSION, "prompt_sha256": prompts[(i, think)]["sha256"], "prompt_tokens_hf": prompts[(i, think)]["n_tokens"],
                   **comp_meta(r), "bucket": p.bucket, "detail": p.detail, "flags": p.flags,
                   "outcome": oc, "outcome_lenient": lo, "bfcl": chk, "decoded": p.calls,
                   "reasoning_chars": len(p.reasoning), "answer": p.answer, "generated": gen}
            done[i] = rec
            fo.write(json.dumps(rec, default=str) + "\n")
            fo.flush()
            print(f"[{a.label}] {'+'.join(sets_of[i])} {i}: {oc} ({p.bucket}) {rec['completion_tokens']} tok {rec['wall_s']}s "
                  f"prompt {rec['prompt_tokens']}/{rec['prompt_tokens_hf']}", flush=True)
    # multi-turn (same resume rule; a budget-aborted entry is not written and restarts from turn 1 next time)
    mt_p = out / "multi_turn.jsonl"
    mt_done = load_done(mt_p)
    with open(mt_p, "a") as fo:
        for i in probe.get("S5", []):
            if i in mt_done:
                continue
            if deadline - time.time() < 90:
                aborted += 1
                continue
            rec = run_multi_turn(a.server, tok, i, think, a.max_tokens, a.ctx, deadline, a.label)
            if rec["outcome"].startswith("budget_abort"):
                aborted += 1
                continue
            mt_done[i] = rec
            fo.write(json.dumps(rec, default=str) + "\n")
            fo.flush()
            print(f"[{a.label}] S5 {i}: {rec['outcome']} steps {rec['steps']} turns {rec['turns_done']}/{rec['turns_gt']} "
                  f"{rec['completion_tokens']} tok {rec['wall_s']}s max prompt {rec['max_prompt_tokens']} "
                  f"max prompt+gen {rec['max_prompt_plus_gen']} buckets {rec['step_buckets']}", flush=True)

    items = [done[i] for i in order if i in done]
    mts = [{k: v for k, v in mt_done[i].items() if k not in ("steps_log", "decoded")} for i in probe.get("S5", []) if i in mt_done]
    def agg(xs):
        d = {}
        for x in xs:
            d[x["outcome"]] = d.get(x["outcome"], 0) + 1
        return d
    summary["single_turn"] = {s: agg([x for x in items if s in x.get("sets", [x["set"]])]) for s in ("S1", "S2", "S3", "S4")}
    summary["single_turn_pace"] = {
        "n": len(items),
        "wall_s_total": round(sum(x["wall_s"] for x in items), 1),
        "completion_tokens_mean": round(sum((x["completion_tokens"] or 0) for x in items) / len(items), 1) if items else None,
        "decode_tok_s_median": sorted(x["decode_tok_s"] or 0 for x in items)[len(items) // 2] if items else None,
        "prompt_tokens_server_eq_hf": sum(1 for x in items if x["prompt_tokens"] == x["prompt_tokens_hf"]),
        "capped": sum(1 for x in items if x["bucket"] == "capped"),
        "python_bool_replies": sum(1 for x in items if "python_bool" in (x.get("flags") or []))}
    summary["multi_turn"] = mts
    summary["multi_turn_outcomes"] = agg(mts)
    summary["transport_errors_this_session"] = TRANSPORT_LOG
    summary["wall_s_this_session"] = round(time.time() - t_start, 1)
    expected = len(order) + len(probe.get("S5", []))
    summary["progress"] = {"done": len(items) + len(mts), "expected": expected, "aborted_this_session": aborted}
    is_complete = len(items) == len(order) and len(mts) == len(probe.get("S5", []))
    (out / ("summary.json" if is_complete else "progress.json")).write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps({k: summary[k] for k in ("label", "single_turn", "multi_turn_outcomes", "progress")}, default=str), flush=True)
    sys.exit(0 if is_complete else 3)


if __name__ == "__main__":
    main()
