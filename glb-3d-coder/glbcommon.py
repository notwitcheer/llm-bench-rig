#!/usr/bin/env python3
"""shared bits for the 91-glb-3d-coder item: config, html extraction, GLB validation (stdlib only)."""
import json, os, re, struct, math
HERE = os.path.dirname(os.path.abspath(__file__))
def cfg(): return json.load(open(os.path.join(HERE, "config.json")))
def prompts(): return json.load(open(os.path.join(HERE, "prompts.json")))["prompts"]
def system_prompt(): return open(os.path.join(HERE, "system.txt")).read()

def extract_html(text):
    """first fenced html block; an unclosed fence runs to the end; else first <!doctype / <html to the end. returns (html|None, how)."""
    if not text: return None, "empty"
    m = re.search(r"```(?:html|HTML)[^\n]*\n(.*?)```", text, re.S)
    if m and m.group(1).strip(): return m.group(1), "fence"
    m = re.search(r"```(?:html|HTML)[^\n]*\n(.*)\Z", text, re.S)
    if m and m.group(1).strip(): return m.group(1), "fence-unclosed"
    m = re.search(r"(<!doctype html|<html)", text, re.I)
    if m: return text[m.start():], "bare"
    return None, "none"

def glb_validate(b):
    """returns (ok, reason, info). magic, version 2, length, JSON chunk, mesh+primitive, POSITION, bufferViews in BIN, finite bbox >0 on >=2 axes."""
    if len(b) < 20: return False, "too short", {}
    magic, ver, ln = struct.unpack("<4sII", b[:12])
    if magic != b"glTF": return False, "bad magic", {}
    if ver != 2: return False, "version %d" % ver, {}
    if ln != len(b): return False, "header length %d != size %d" % (ln, len(b)), {}
    off = 12; js = None; binlen = 0
    while off + 8 <= len(b):
        cl, ct = struct.unpack("<I4s", b[off:off+8]); body = b[off+8:off+8+cl]
        if len(body) != cl: return False, "chunk overruns file", {}
        if ct == b"JSON": 
            try: js = json.loads(body.decode("utf-8").rstrip("\x00 "))
            except Exception as e: return False, "json chunk: %s" % str(e)[:60], {}
        elif ct == b"BIN\x00": binlen = cl
        off += 8 + cl
    if js is None: return False, "no JSON chunk", {}
    if str(js.get("asset", {}).get("version")) != "2.0": return False, "asset.version", {}
    meshes = js.get("meshes") or []
    prims = [p for m in meshes for p in (m.get("primitives") or [])]
    if not prims: return False, "no mesh primitive", {}
    acc = js.get("accessors") or []; bvs = js.get("bufferViews") or []
    for i, bv in enumerate(bvs):
        if bv.get("byteOffset", 0) + bv.get("byteLength", 0) > binlen: return False, "bufferView %d outside BIN" % i, {}
    lo = [math.inf]*3; hi = [-math.inf]*3; verts = 0
    for p in prims:
        pi = (p.get("attributes") or {}).get("POSITION")
        if pi is None or pi >= len(acc): return False, "primitive without POSITION", {}
        a = acc[pi]
        if a.get("count", 0) <= 0: return False, "POSITION count 0", {}
        verts += a["count"]
        mn, mx = a.get("min"), a.get("max")
        if not mn or not mx or len(mn) < 3: return False, "POSITION min/max missing", {}
        for k in range(3):
            if not (math.isfinite(mn[k]) and math.isfinite(mx[k])): return False, "POSITION not finite", {}
            lo[k] = min(lo[k], mn[k]); hi[k] = max(hi[k], mx[k])
    ext = [hi[k]-lo[k] for k in range(3)]
    if sum(1 for e in ext if e > 1e-9) < 2: return False, "degenerate bbox %s" % [round(e, 4) for e in ext], {}
    info = {"meshes": len(meshes), "primitives": len(prims), "vertices": verts, "nodes": len(js.get("nodes") or []), "materials": len(js.get("materials") or []),
            "animations": len(js.get("animations") or []), "bbox_extent": [round(e, 3) for e in ext], "bytes": len(b)}
    return True, "ok", info
