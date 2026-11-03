#!/usr/bin/env python3
"""clover_trace_render_v2.py <err.txt> <out.mp4> [--ffmpeg PATH] [--first 4] [--last 23] [--fps 24] [--png-dir DIR --png-at t1,t2,..]

v2 of the Clover decode-trace clip, vertical 1080x1350, for phone size. SAME data as v1: the CLOVERTRACE json lines
(per verify step: pos, replay, draft[], out[], accepted, t_us) printed by llama-speculative-simple with the local
CLOVER_TRACE patch. nothing is invented: every token on screen is a token in the log.

what it draws, per non-replay verify step (a 'block'):
  land    the whole drafted block appears at once as a yellow-tinted box (token boundaries ticked), counter 'kept ? of N'.
  resolve accepted tokens flash green then turn white, the bonus token (the causal pass's own token) is cyan,
          rejected draft tokens flash red with a strikethrough and vanish; counter 'kept K of N' + running total.
  settle  committed text only, the view scrolls to the next block.
  replay  a replay step (checkpoint re-verify after a reject) only commits one more token; shown as a short cyan '+1' beat.
the opener is a close-up on the first block landing, then the camera pulls back to the last ~10 lines.
the timeline is slowed; the slowdown factor is COMPUTED from the log's own t_us deltas over the shown stretch and printed
on screen with the real decode speed (read from the llama.cpp 'decoded N tokens' line in err.txt). no side-by-side
second model; there is no measured one-token-per-step run, so none is drawn. not a denoising-step capture: the llama.cpp
path drafts a block in ONE bidirectional pass, so there are no intermediate noise states.
"""
import json, sys, os, re, subprocess, math
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageFont

A = sys.argv
src, out = A[1], A[2]
def opt(name, default):
    return A[A.index(name) + 1] if name in A else default
FF = opt("--ffmpeg", "ffmpeg"); FIRST = int(opt("--first", 4)); LAST = int(opt("--last", 23)); FPS = int(opt("--fps", 24))
PNG_DIR = opt("--png-dir", None); PNG_AT = [float(x) for x in opt("--png-at", "").split(",") if x]

steps = []; real_tps = None
for l in open(src, errors="replace"):
    if "CLOVERTRACE " in l:
        try: steps.append(json.loads(l.split("CLOVERTRACE ", 1)[1]))
        except Exception: pass
    m = re.search(r"decoded\s+(\d+) tokens in\s+([\d.]+) seconds, speed:\s+([\d.]+)", l)
    if m: real_tps = float(m.group(3)); real_tok = int(m.group(1)); real_sec = float(m.group(2))
if not steps: print("no CLOVERTRACE lines"); sys.exit(2)
N = len(steps)

# ---- token log -> committed text + per-block descriptors ---------------------------------------------------------
toks = {}
def committed(upto): return [(toks[i], "c") for i in range(upto) if i in toks]
for s in steps:
    for i, t in enumerate(s["out"]): toks[s["pos"] + i] = t
tot_acc_all = sum(s["accepted"] for s in steps if not s["replay"]); tot_dr_all = sum(len(s["draft"]) for s in steps if not s["replay"])

# ---- layout ---------------------------------------------------------------------------------------------------------
W, H = 1080, 1350
PANEL_Y, PANEL_H = 252, 748
SS = 2                      # supersample for crisp zoom
FONT_PX = 56; COLS = 25; NV = 10; LH = 72; PAD_TOP = 14
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"; MONOB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SANSB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"; SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
def font(p, px): return ImageFont.truetype(p, px)
FCODE = font(MONO, FONT_PX * SS); ADV = FCODE.getlength("M") / SS
FNL = font(MONO, int(FONT_PX * 0.8) * SS)
X0 = (W - COLS * ADV) / 2

BG = (14, 16, 22); WHITE = (238, 238, 242); DIM = (150, 156, 170)
YEL_BG = (140, 110, 8); YEL_LN = (255, 208, 40); YEL_TX = (255, 238, 150)
GRN_HI = (46, 220, 100); GRN_LO = (32, 130, 70); GRN_TX = (140, 255, 170)
RED_BG = (205, 38, 52); RED_TX = (255, 232, 232)
CY_BG = (40, 214, 238); CY_TX = (110, 235, 255); CY_DK = (4, 22, 28)

def lerp(a, b, t): return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
def sstep(t): t = max(0.0, min(1.0, t)); return t * t * (3 - 2 * t)

def layout(prefix, block):
    """prefix: committed token list [(text,'c')]. block: list of (token_text, state). wraps on TOKEN boundaries (a token that would
    cross the right edge moves whole to the next row). returns cells [(row, col, ch, state, tokidx)], rows."""
    cells = []; r = 0; c = 0
    seq = [(t, st, None) for t, st in prefix] + [(t, st, ti) for ti, (t, st) in enumerate(block)]
    for t, st, ti in seq:
        if "\n" not in t and c > 0 and c + len(t) > COLS and len(t) <= COLS: r += 1; c = 0
        for ch in t:
            if ch == "\n":
                if st != "c": cells.append((r, min(c, COLS - 1), "↵", st, ti))
                r += 1; c = 0; continue
            if c == 0 and ch == " " and cells and cells[-1][0] < r: continue
            if c >= COLS: r += 1; c = 0
            cells.append((r, c, ch, st, ti)); c += 1
    return cells, r + 1

# ---- per-block descriptors for the shown stretch ---------------------------------------------------------------
blocks = []   # dict per non-replay step in [FIRST, LAST]
run_acc = 0; run_dr = 0
for k in range(FIRST, LAST + 1):
    s = steps[k - 1]
    if s["replay"] or not s["draft"]: continue
    nxt = steps[k] if k < N else None
    rep = nxt if (nxt and nxt["replay"]) else None
    pre = committed(s["pos"]); dr = s["draft"]; o = s["out"]; acc = s["accepted"]
    bonus = o[acc:]               # the causal pass's own token(s) after the kept run (1 token)
    rej = dr[acc:]
    draft_cells, r1 = layout(pre, [(t, "draft") for t in dr])
    res_blk = [(t, "acc") for t in o[:acc]] + [(t, "bonus") for t in bonus] + [(t, "rej") for t in rej]
    res_cells, r2 = layout(pre, res_blk)
    after = committed(s["pos"] + len(o))
    settle_cells, r3 = layout(pre, [(t, "acc") for t in o[:acc]] + [(t, "bonus") for t in bonus])
    rep_cells = None; r4 = r3; rep_bonus_tok = None
    if rep:
        rep_bonus_tok = rep["out"][-1]
        rep_cells, r4 = layout(after, [(rep_bonus_tok, "bonus")])
        after2 = committed(rep["pos"] + len(rep["out"]))
        rep_settle, r5 = layout(after2, [])
    run_acc += acc; run_dr += len(dr)
    blocks.append(dict(k=k, n=len(dr), acc=acc, nrej=len(rej), nb=len(bonus), draft=draft_cells, res=res_cells, settle=settle_cells,
                       rep=rep_cells, rep_tok=rep_bonus_tok, rows_ld=max(r1, r2), rows_s=r3, rows_rep=r4,
                       run_acc=run_acc, run_dr=run_dr, prev_acc=run_acc - acc, prev_dr=run_dr - len(dr), has_rep=bool(rep)))
if not blocks: print("empty stretch"); sys.exit(2)

# ---- timeline ----------------------------------------------------------------------------------------------------
T_LAND, T_RES, T_SET, T_REP = 0.85, 1.25, 0.28, 0.45
T_LAND0, T_RES0 = 1.0, 1.9          # the opener holds longer
T_OUTRO = 3.2
timeline = []; t = 0.0
for bi, b in enumerate(blocks):
    tl = T_LAND0 if bi == 0 else T_LAND; tr = T_RES0 if bi == 0 else T_RES
    for ph, d in (("land", tl), ("resolve", tr), ("settle", T_SET)) + ((("replay", T_REP),) if b["has_rep"] else ()):
        timeline.append((t, t + d, bi, ph)); t += d
t_stretch = t
timeline.append((t, t + T_OUTRO, len(blocks) - 1, "outro")); t += T_OUTRO
DUR = t

# real time of the shown stretch from the log's own timestamps
k_first = blocks[0]["k"]; k_end = blocks[-1]["k"] + (1 if blocks[-1]["has_rep"] else 0)
t_us0 = steps[k_first - 2]["t_us"] if k_first >= 2 else steps[0]["t_us"]
real_stretch = (steps[k_end - 1]["t_us"] - t_us0) / 1e6
SLOW = t_stretch / real_stretch
stretch_tokens = sum(len(steps[i - 1]["out"]) for i in range(k_first, k_end + 1))
SLOW_TXT = f"{int(round(SLOW))}x"
REAL_TPS = real_tps if real_tps else 0.0

OUTRO_CELLS, OUTRO_ROWS = layout(committed(max(toks) + 1), [])
# camera: close-up on block 0 (panel coordinates)
def block_center(cells_):
    rows = [c[0] for c in cells_ if c[3] != "c"]
    return rows

def frame_state(ts):
    ts = min(ts, DUR - 1e-6)
    for (a, bnd, bi, ph) in timeline:
        if a <= ts < bnd: return bi, ph, ts - a, (ts - a) / (bnd - a)
    return len(blocks) - 1, "outro", 0.0, 1.0

def anchor_for(bi, ph, f):
    b = blocks[bi]
    A0 = b["rows_ld"]
    nxt = blocks[bi + 1]["rows_ld"] if bi + 1 < len(blocks) else b["rows_rep"]
    if ph in ("land", "resolve"): return float(A0)
    if ph == "settle": return A0 + (nxt - A0) * sstep(f)
    return float(nxt)

def draw_cells(d, cells, anchor, bi, ph, u, rej_alpha=1.0):
    """draw code layer at supersample scale into d."""
    b = blocks[bi]
    first_row = anchor - NV
    # block outlines per row
    rows_cols = {}
    for (r, c, ch, st, ti) in cells:
        if st != "c": rows_cols.setdefault(r, [10 ** 9, -1, set()]); e = rows_cols[r]; e[0] = min(e[0], c); e[1] = max(e[1], c); e[2].add(st)
    for (r, c, ch, st, ti) in cells:
        y = (PAD_TOP + (r - first_row) * LH)
        if y < -LH or y > PANEL_H: continue
        x = X0 + c * ADV
        bg = None; tx = WHITE; strike = False
        if st == "draft": bg = lerp((222, 182, 24), YEL_BG, sstep(u / 0.3)) if ph == "land" else YEL_BG; tx = lerp((24, 18, 0), YEL_TX, sstep(u / 0.3)) if ph == "land" else YEL_TX
        elif st == "acc":
            if ph == "resolve":
                fl = sstep(u / 0.45); bg = lerp(GRN_HI, GRN_LO, fl); tx = WHITE
            elif ph == "settle": bg = lerp(GRN_LO, BG, sstep(u / T_SET)); tx = WHITE
            else: bg = None
        elif st == "bonus":
            tx = CY_DK; bg = CY_BG
            if ph == "settle": bg = lerp(CY_BG, BG, sstep(u / T_SET)); tx = lerp(CY_DK, WHITE, sstep(u / T_SET))
        elif st == "rej":
            fade = 1.0
            if ph == "resolve" and u > 0.95: fade = 1 - sstep((u - 0.95) / 0.25)
            if fade <= 0.01: continue
            bg = lerp(BG, RED_BG, fade); tx = lerp(BG, RED_TX, fade); strike = True
        if bg is not None:
            d.rectangle(((x) * SS, y * SS, (x + ADV) * SS + 1, (y + LH) * SS), fill=bg)
        if st == "draft" and ti is not None:
            tstart = [cc for cc in cells if cc[3] == "draft" and cc[4] == ti][0]
            if tstart[0] == r and tstart[1] == c and ti > 0: d.rectangle((x * SS, y * SS + 6, x * SS + 4, (y + LH) * SS - 6), fill=BG)
        if ch == "↵":
            d.text(((x + ADV * 0.02) * SS, (y + 8) * SS), "↵", font=FNL, fill=tx)
        else:
            d.text((x * SS, (y + 4) * SS), ch, font=FCODE, fill=tx)
        if strike: d.rectangle((x * SS, (y + LH * 0.5) * SS - 3, (x + ADV) * SS, (y + LH * 0.5) * SS + 4), fill=lerp(BG, (255, 255, 255), fade))
    # one outline around the whole active block (rectilinear polygon over the row spans)
    if ph in ("land", "resolve") and rows_cols:
        col = (lerp((255, 255, 255), YEL_LN, sstep(u / 0.3)) if ph == "land" else lerp(YEL_LN, BG, sstep(u / 0.4)))
        rs = sorted(rows_cols); L = {r: (X0 + rows_cols[r][0] * ADV) * SS for r in rs}; R = {r: (X0 + (rows_cols[r][1] + 1) * ADV) * SS for r in rs}
        top = {r: (PAD_TOP + (r - first_row) * LH) * SS for r in rs}; bot = {r: top[r] + LH * SS for r in rs}
        pts = []
        for r in rs: pts += [(R[r], top[r]), (R[r], bot[r])]
        for r in reversed(rs): pts += [(L[r], bot[r]), (L[r], top[r])]
        d.line(pts + [pts[0]], fill=col, width=7, joint="curve")

FHEAD = font(SANSB, 52); FLEG = font(SANSB, 33); FSUB = font(SANS, 29); FCAP = font(SANS, 26); FCAPB = font(SANSB, 29)
FPL = font(MONOB, 28); FBIG = font(MONOB, 92); FTOT = font(MONOB, 40); FBON = font(MONOB, 54); FSPD = font(SANSB, 34)

def render(i):
    ts = i / FPS
    bi, ph, u, f = frame_state(ts)
    b = blocks[bi]
    im = Image.new("RGB", (W, H), BG); dr = ImageDraw.Draw(im)
    # header
    dr.text((40, 14), "writes a whole block at once,", font=FHEAD, fill=WHITE)
    dr.text((40, 76), "then keeps what it believes", font=FHEAD, fill=YEL_TX)
    x = 40; y = 156
    for lab, bg, tx in (("drafted", YEL_LN, YEL_TX), ("kept", GRN_HI, GRN_TX), ("rejected", (235, 60, 70), (255, 120, 130)), ("bonus token", (30, 200, 230), CY_TX)):
        dr.rounded_rectangle((x, y + 4, x + 34, y + 38), radius=6, fill=bg)
        dr.text((x + 46, y), lab, font=FLEG, fill=tx); x += 46 + int(FLEG.getlength(lab)) + 38
    dr.text((40, 200), f"real speed {REAL_TPS:.0f} tok/s  |  shown slowed ~{SLOW_TXT}", font=FSPD, fill=(255, 255, 255))
    # code panel (supersampled, camera)
    if ph == "outro":
        cells = OUTRO_CELLS; anchor = float(OUTRO_ROWS)
        ph_c = "outro"
    else:
        anchor = anchor_for(bi, ph, f)
        if ph == "land": cells = b["draft"]
        elif ph == "resolve": cells = b["res"]
        elif ph == "settle": cells = b["settle"]
        else: cells = b["rep"]
        ph_c = ph
    layer = Image.new("RGB", (W * SS, PANEL_H * SS), BG); ld = ImageDraw.Draw(layer)
    draw_cells(ld, cells, anchor, bi, ph_c if ph_c != "outro" else "settle", u)
    # camera
    Z0 = 1.15
    if bi == 0 and ph in ("land", "resolve", "settle") or (bi == 1 and ph == "land"):
        tcam = ts
        zf = 1 - sstep((tcam - (T_LAND0 + 0.9)) / 1.2)
    else: zf = 0.0
    Z = 1 + (Z0 - 1) * zf
    rows0 = [c[0] for c in blocks[0]["draft"] if c[3] == "draft"]
    first_row0 = blocks[0]["rows_ld"] - NV
    cy0 = PAD_TOP + ((min(rows0) + max(rows0)) / 2 + 0.5 - first_row0) * LH
    ccx = W / 2; ccy = PANEL_H / 2 + (cy0 - PANEL_H / 2) * zf
    vw, vh = W / Z, PANEL_H / Z
    bx0 = max(0, min(W - vw, ccx - vw / 2)); by0 = max(0, min(PANEL_H - vh, ccy - vh / 2))
    pnl = layer.resize((W, PANEL_H), Image.LANCZOS, box=(bx0 * SS, by0 * SS, (bx0 + vw) * SS, (by0 + vh) * SS))
    fade = Image.new("L", (W, 90), 0); fd_ = ImageDraw.Draw(fade)
    for yy in range(90): fd_.line((0, yy, W, yy), fill=int(255 * (1 - yy / 90) ** 1.3))
    pnl.paste(Image.new("RGB", (W, 90), BG), (0, 0), fade)
    im.paste(pnl, (0, PANEL_Y))
    dr.line((0, PANEL_Y, W, PANEL_Y), fill=(40, 46, 60), width=2); dr.line((0, PANEL_Y + PANEL_H, W, PANEL_Y + PANEL_H), fill=(40, 46, 60), width=2)
    # bottom: strip + counters
    n = b["n"]; gap = 6; cw = (1000 - n * gap) / (n + 1)
    sy = PANEL_Y + PANEL_H + 12
    def cell_col(j):
        if ph == "land": return YEL_LN
        if ph == "outro" or ph in ("settle", "replay"):
            return (36, 150, 74) if j < b["acc"] else (150, 38, 50)
        if j < b["acc"]: return lerp(GRN_HI, (36, 150, 74), sstep(u / 0.6))
        return lerp((235, 60, 70), (150, 38, 50), sstep((u - 0.7) / 0.3))
    for j in range(n if ph != "outro" else 0):
        cx_ = 40 + j * (cw + gap)
        dr.rounded_rectangle((cx_, sy, cx_ + cw, sy + 40), radius=6, fill=cell_col(j))
        if j >= b["acc"] and ph != "land": dr.line((cx_ + 6, sy + 6, cx_ + cw - 6, sy + 34), fill=(30, 10, 12), width=3); dr.line((cx_ + 6, sy + 34, cx_ + cw - 6, sy + 6), fill=(30, 10, 12), width=3)
    if ph == "outro":
        gw = 1000 * tot_acc_all / tot_dr_all
        dr.rounded_rectangle((40, sy, 40 + gw - 3, sy + 40), radius=6, fill=(36, 160, 78)); dr.rounded_rectangle((40 + gw + 3, sy, 1040, sy + 40), radius=6, fill=(190, 44, 58))
        dr.text((52, sy + 3), f"kept {100*tot_acc_all/tot_dr_all:.0f}%", font=FPL, fill=(4, 30, 14)); dr.text((52 + gw, sy + 3), f"rejected {100-100*tot_acc_all/tot_dr_all:.0f}%", font=FPL, fill=(255, 235, 235))
    cx_ = 40 + n * (cw + gap)
    if ph == "outro": cx_ = 5000
    dr.rounded_rectangle((cx_, sy, cx_ + cw, sy + 40), radius=6, fill=CY_BG if ph != "land" else (30, 38, 52))
    tl_ = "+1"
    if ph != "outro": dr.text((cx_ + cw / 2 - FPL.getlength(tl_) / 2, sy + 4), tl_, font=FPL, fill=(CY_DK if ph != "land" else (120, 130, 150)))
    # counter
    cy_ = sy + 46
    if ph == "outro":
        pf = sstep(u / 0.5) if False else 1.0
        dr.text((40, cy_), f"kept {tot_acc_all} of {tot_dr_all}", font=FBIG, fill=GRN_TX)
        dr.text((40, cy_ + 104), f"whole trace: {N} steps, {real_tok} tokens", font=FTOT, fill=WHITE)
    else:
        if ph == "land":
            dr.text((40, cy_), f"drafted {n} at once", font=FBIG, fill=YEL_TX)
            ra, rd = b["prev_acc"], b["prev_dr"]
        else:
            pulse = 1.0
            col = GRN_TX if ph in ("resolve", "settle", "replay") else WHITE
            dr.text((40, cy_), f"kept {b['acc']} of {n}", font=FBIG, fill=col)
            tb = 40 + int(FBIG.getlength(f"kept {b['acc']} of {n}")) + 24
            ra, rd = b["run_acc"], b["run_dr"]
            dr.text((tb, cy_ + 10), f"+{b['nb']}", font=FBON, fill=CY_TX); dr.text((tb, cy_ + 66), "bonus", font=FLEG, fill=CY_TX)
        pct = (100 * ra / rd) if rd else 0
        if rd: dr.text((40, cy_ + 104), f"running total: {ra} of {rd} kept ({pct:.0f}%)", font=FTOT, fill=WHITE)
    # caption
    cap_y = H - 128
    dr.text((40, cap_y), "Clover-35B-A3B on one RTX 5090 through Hermes Agent,", font=FCAPB, fill=(210, 214, 224))
    dr.text((40, cap_y + 33), "self-speculative block drafting (llama.cpp PR build)", font=FCAPB, fill=(210, 214, 224))
    dr.text((40, cap_y + 70), "trace: msm-v2 prompt, real draft/verify log, token for token", font=FCAPB, fill=(190, 196, 208))
    return im

def to_bytes(i): return render(i).tobytes()

if __name__ == "__main__":
    total = int(math.ceil(DUR * FPS))
    print(f"blocks {len(blocks)} steps {FIRST}-{LAST} dur {DUR:.1f}s frames {total} real_stretch {real_stretch*1000:.0f} ms slow {SLOW:.1f}x "
          f"stretch tokens {stretch_tokens} stretch kept {blocks[-1]['run_acc']} of {blocks[-1]['run_dr']} full trace kept {tot_acc_all} of {tot_dr_all} real {REAL_TPS}", flush=True)
    if PNG_DIR:
        os.makedirs(PNG_DIR, exist_ok=True)
        for tt in PNG_AT: render(int(tt * FPS)).save(f"{PNG_DIR}/t{tt:05.2f}.png")
        sys.exit(0)
    cmd = [FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-framerate", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(10) as pool:
        for b_ in pool.imap(to_bytes, range(total), chunksize=4): p.stdin.write(b_)
    p.stdin.close(); rc = p.wait()
    print("ffmpeg rc", rc, "size", os.path.getsize(out) if os.path.exists(out) else None)
    sys.exit(rc)
