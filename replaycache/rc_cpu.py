#!/usr/bin/env python3
"""rc_cpu.py: the CPU legs. every llama-server here runs with CUDA_VISIBLE_DEVICES= and -ngl 0 (the item starts them); this script only talks HTTP.
  embed  : entries (description + build question) and every query -> best entry + cosine s per query            -> <run>/cpu/B-embed.json
  B      : per query, the 1B extracts the schema params for its BEST entry (every query, so any threshold can be scored afterwards), solve() runs
           in an isolated child                                                                                 -> <run>/cpu/B/<key>.json
  C      : the 1B alone on the same question: C = brief reasoning + '####' (primary control), Cd = the number only (secondary)   -> <run>/cpu/C/<key>.json
rc: 0 done | 3 deadline | 1 errors"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rc_common import *

ap = argparse.ArgumentParser()
ap.add_argument("leg", choices=["embed", "B", "C"])
ap.add_argument("--run", required=True)
ap.add_argument("--sel", required=True)
ap.add_argument("--entries", required=True)
ap.add_argument("--emb-base", default="http://127.0.0.1:8312")
ap.add_argument("--lm-base", default="http://127.0.0.1:8311")
ap.add_argument("--deadline", type=float, default=0)
ap.add_argument("--limit-templates", type=int, default=0)
ap.add_argument("--limit-items", type=int, default=0)
a = ap.parse_args()

sel = json.load(open(a.sel))
LT = a.limit_templates or None
CPU = os.path.join(a.run, "cpu")
os.makedirs(CPU, exist_ok=True)
QP, DP = "task: search result | query: ", "title: none | text: "


def items():
    return limited_items(sel, LT, a.limit_items)


def late():
    return a.deadline and time.time() > a.deadline


def load_entries():
    ents = []
    for f in sorted(os.listdir(a.entries)):
        if f.endswith(".json") and not f.endswith(".failed.json"):
            e = json.load(open(os.path.join(a.entries, f)))
            e["code_path"] = os.path.join(a.entries, e["code_file"])
            ents.append(e)
    return ents


def embed(texts):
    r = post_json(a.emb_base + "/v1/embeddings", {"model": "embed", "input": texts}, 600)
    out = []
    for d in sorted(r["data"], key=lambda d: d["index"]):
        v = d["embedding"]; n = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / n for x in v])
    return out


def dot(u, v):
    return sum(x * y for x, y in zip(u, v))


if a.leg == "embed":
    ents = load_entries()
    if not ents:
        log("no entries"); sys.exit(1)
    t0 = time.time()
    ev = embed([DP + entry_text(e) for e in ents])
    its = items()
    rows = []
    for (kind, tid, inst) in its:
        q = row_for(kind, tid, inst)["question"]
        t1 = time.time(); qv = embed([QP + q])[0]; ms = (time.time() - t1) * 1000
        sc = [dot(qv, v) for v in ev]
        j = max(range(len(sc)), key=lambda i: sc[i])
        rows.append({"key": item_key(kind, tid, inst), "kind": kind, "tid": tid, "inst": inst, "s": sc[j], "entry": ents[j]["tid"], "embed_ms": round(ms, 1),
                     "own_entry_s": next((sc[i] for i, e in enumerate(ents) if e["tid"] == tid), None)})
    atomic_json(os.path.join(CPU, "B-embed.json"), {"entries": [e["tid"] for e in ents], "rows": rows, "doc_prefix": DP, "query_prefix": QP})
    log("embed: %d entries, %d queries, %.1f s" % (len(ents), len(rows), time.time() - t0))
    sys.exit(0)

if a.leg == "B":
    ents = {e["tid"]: e for e in load_entries()}
    emb = json.load(open(os.path.join(CPU, "B-embed.json")))
    pending = 0
    for r in emb["rows"]:
        if late():
            pending += 1; continue
        path = os.path.join(CPU, "B", r["key"] + ".json")
        if os.path.exists(path):
            continue
        e = ents[r["entry"]]
        q = row_for(r["kind"], r["tid"], r["inst"])["question"]
        keys = "\n".join('- "%s": %s' % (s["name"], s["description"]) for s in e["schema"])
        prompt = EXTRACT_PROMPT.format(description=e["description"], build_question=e["build_question"], example_params=json.dumps(e["build_params"]), keys=keys, question=q)
        props = {s["name"]: {"type": "number"} for s in e["schema"]}
        t0 = time.time()
        try:
            resp = chat(a.lm_base, [{"role": "user", "content": prompt}], 400, model="llama1b", temperature=0, seed=1, cache_prompt=False,
                        response_format={"type": "json_schema", "json_schema": {"name": "params", "strict": True, "schema": {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}}})
            txt, _ = msg_of(resp)
            toks = resp.get("usage", {}).get("completion_tokens", 0); ptoks = resp.get("usage", {}).get("prompt_tokens", 0)
            try:
                params = json.loads(txt)
                okp = isinstance(params, dict) and all(isinstance(params.get(n), (int, float)) and not isinstance(params.get(n), bool) for n in props)
            except Exception:
                params, okp = None, False
        except Exception as ex:
            log("B error", r["key"], repr(ex)[:100]); params, okp, toks, ptoks, txt = None, False, 0, 0, ""
        t1 = time.time()
        val, why = None, ""
        if okp:
            ok, val, why = run_solve(e["code_path"], {n: params[n] for n in props})
        else:
            why = "missing or non-number param"
        wall = time.time() - t0
        row = row_for(r["kind"], r["tid"], r["inst"]); g = golden_of(row)
        rec = {"key": r["key"], "kind": r["kind"], "tid": r["tid"], "inst": r["inst"], "entry": r["entry"], "s": r["s"], "params": params, "extract_ok": okp, "value": val, "served": val is not None,
               "why": why, "golden": g, "correct": same(val, g), "extract_ms": round((t1 - t0) * 1000, 1), "solve_ms": round((time.time() - t1) * 1000, 1), "embed_ms": r["embed_ms"],
               "cpu_ms": round(wall * 1000 + r["embed_ms"], 1), "lm_tokens": toks, "lm_prompt_tokens": ptoks}
        atomic_json(path, rec)
        log("B %s s=%.3f entry=%d value=%s gold=%s %s %.0f ms" % (r["key"], r["s"], r["entry"], val, g, "ok" if rec["correct"] else "wrong/miss", rec["cpu_ms"]))
    log("B done, %d pending (deadline)" % pending)
    sys.exit(3 if pending else 0)

if a.leg == "C":
    its = items()
    pending = 0
    for (kind, tid, inst) in its:
        if late():
            pending += 1; continue
        key = item_key(kind, tid, inst)
        path = os.path.join(CPU, "C", key + ".json")
        if os.path.exists(path):
            continue
        row = row_for(kind, tid, inst); g = golden_of(row); out = {"key": key, "kind": kind, "tid": tid, "inst": inst, "golden": g}
        for tag, prompt, mt in (("C", C_COT_PROMPT, 700), ("Cd", C_DIRECT_PROMPT, 24)):
            t0 = time.time()
            try:
                r = chat(a.lm_base, [{"role": "user", "content": prompt.format(question=row["question"])}], mt, model="llama1b", temperature=0, seed=1, cache_prompt=False)
                txt, _ = msg_of(r)
                v, how = parse_final(txt)
                out[tag] = {"answer": v, "how": how, "correct": same(v, g), "tokens": r.get("usage", {}).get("completion_tokens", 0), "wall_ms": round((time.time() - t0) * 1000, 1), "finish": r["choices"][0].get("finish_reason")}
            except Exception as ex:
                out[tag] = {"answer": None, "how": "error", "correct": False, "tokens": 0, "wall_ms": round((time.time() - t0) * 1000, 1), "error": repr(ex)[:100]}
        atomic_json(path, out)
        log("C %s C=%s Cd=%s gold=%s" % (key, out["C"]["answer"], out["Cd"]["answer"], g))
    log("C done, %d pending (deadline)" % pending)
    sys.exit(3 if pending else 0)
