#!/usr/bin/env python3
"""clover_trace_render_v3.py <err.txt> <out.mp4> [--ffmpeg PATH] [--first 4] [--last 23] [--fps 24] [--png-dir DIR --png-at t1,t2,..]

v3 of the Clover decode-trace clip: desktop 16:9, 1920x1080 (operator rule 2026-11-03: X/GPU videos are 16:9, 4:5 is for
mobile-app videos only). SAME data as v1/v2: the CLOVERTRACE json lines (per verify step: pos, replay, draft[], out[],
accepted, t_us) printed by llama-speculative-simple with the local CLOVER_TRACE patch. nothing is invented: every token on
screen is a token in the log.

layout: code panel on the left (37 columns, 52 px type), a stats column on the right (legend, block counter, big
'kept K of N', a padded '+1 bonus' chip, 'kept so far', speed line), a full-width block bar under the panel that fills left to
right as the blocks resolve (one segment per block: green kept, red rejected, cyan bonus; 'kept/drafted' under each), caption last.

per block (non-replay verify step):
  land     the whole drafted block appears at once as a bright yellow box (token boundaries ticked), counter 'drafted N'.
  resolve  accepted tokens flash green, the bonus token (the causal pass's own token) is a padded cyan chip, rejected draft
           tokens flash red with a strikethrough and then vanish; the block's segment in the bar fills in.
  settle   committed text only. the replay step that follows a reject (checkpoint re-verify, one more token) is not drawn as a
           beat any more: its token just appears as committed text.
the end card keeps the full bar and quotes per-block tallies only. no whole-trace 'accepted' total is shown: llama.cpp's own
end counter in err.txt (288 of 288) disagrees with the per-step tally (266 of 453) and that is unresolved. 'kept so far' is the
running sum of the per-block 'kept' tallies over the blocks shown, labelled as such.
the slowdown factor is COMPUTED from the log's own t_us deltas over the shown stretch; the real decode speed is read from the
llama.cpp 'decoded N tokens' line in err.txt (whole trace leg).
"""
import json, sys, os, re, subprocess, math
from multiprocessing import Pool
from PIL import Image, ImageDraw, ImageFont

A = sys.argv
src, out = A[1], A[2]
def opt(name, default):
    return A[A.index(name) + 1] if name in A else default
FF = opt("--ffmpeg", "ffmpeg"); FIRST = int(opt("--first", 4)); LAST = int(opt("--last", 23)); FPS = int(opt("--fps", 24))
POOL = int(opt("--pool", 8))
PNG_DIR = opt("--png-dir", None); PNG_AT = [float(x) for x in opt("--png-at", "").split(",") if x]

steps = []; real_tps = None; real_tok = 0
for l in open(src, errors="replace"):
    if "CLOVERTRACE " in l:
        try: steps.append(json.loads(l.split("CLOVERTRACE ", 1)[1]))
        except Exception: pass
    m = re.search(r"decoded\s+(\d+) tokens in\s+([\d.]+) seconds, speed:\s+([\d.]+)", l)
    if m: real_tps = float(m.group(3)); real_tok = int(m.group(1))
if not steps: print("no CLOVERTRACE lines"); sys.exit(2)
N = len(steps)

toks = {}
def committed(upto): return [(toks[i], "c") for i in range(upto) if i in toks]
for s in steps:
    for i, t in enumerate(s["out"]): toks[s["pos"] + i] = t

# ---- layout -----------------------------------------------------------------------------------------------------------
W, H = 1920, 1080
PX, PY, PW, PH = 40, 146, 1160, 640          # code panel on the frame
SS = 2
FONT_PX = 52; LH = 66; PAD_TOP = 8
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"; MONOB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SANSB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"; SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
def font(p, px): return ImageFont.truetype(p, px)
FCODE = font(MONO, FONT_PX * SS); ADV = FCODE.getlength("M") / SS
COLS = int((PW - 24) // ADV); NV = int(PH // LH)
FNL = font(MONO, int(FONT_PX * 0.8) * SS)
X0 = (PW - COLS * ADV) / 2

BG = (14, 16, 22); WHITE = (238, 238, 242); DIM = (178, 185, 202)
YEL_BG = (255, 214, 10); YEL_LN = (255, 214, 10); YEL_TX = (255, 244, 170); YEL_INK = (22, 16, 0)    # drafted: the legend swatch colour exactly, dark ink on it
YEL_SW = YEL_LN
GRN_HI = (46, 220, 100); GRN_LO = (32, 130, 70); GRN_TX = (140, 255, 170)
RED_BG = (205, 38, 52); RED_TX = (255, 232, 232)
CY_BG = (40, 214, 238); CY_TX = (110, 235, 255); CY_DK = (4, 22, 28)

def lerp(a, b, t): return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))
def sstep(t): t = max(0.0, min(1.0, t)); return t * t * (3 - 2 * t)

def layout(prefix, block):
    """prefix: committed token list [(text,'c')]. block: list of (token_text, state). wraps on TOKEN boundaries."""
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

# ---- per-block descriptors for the shown stretch -----------------------------------------------------------------------
blocks = []
run_acc = 0; run_dr = 0
for k in range(FIRST, LAST + 1):
    s = steps[k - 1]
    if s["replay"] or not s["draft"]: continue
    nxt = steps[k] if k < N else None
    rep = nxt if (nxt and nxt["replay"]) else None
    pre = committed(s["pos"]); dr = s["draft"]; o = s["out"]; acc = s["accepted"]
    bonus = o[acc:]; rej = dr[acc:]
    draft_cells, r1 = layout(pre, [(t, "draft") for t in dr])
    res_blk = [(t, "acc") for t in o[:acc]] + [(t, "bonus") for t in bonus] + [(t, "rej") for t in rej]
    res_cells, r2 = layout(pre, res_blk)
    set_blk = [(t, "acc") for t in o[:acc]] + [(t, "bonus") for t in bonus]
    if rep: set_blk += [(rep["out"][-1], "c")]      # the replay token just appears as committed text
    settle_cells, r3 = layout(pre, set_blk)
    run_acc += acc; run_dr += len(dr)
    blocks.append(dict(k=k, n=len(dr), acc=acc, nrej=len(rej), nb=len(bonus), draft=draft_cells, res=res_cells, settle=settle_cells,
                       rows_ld=max(r1, r2), rows_s=r3, run_acc=run_acc, run_dr=run_dr, has_rep=bool(rep)))
if not blocks: print("empty stretch"); sys.exit(2)
NB = len(blocks)

# ---- timeline ----------------------------------------------------------------------------------------------------------
T_LAND, T_RES, T_SET = 0.85, 1.25, 0.28
T_LAND0, T_RES0 = 1.0, 1.9
T_OUTRO = 4.2
timeline = []; t = 0.0
for bi, b in enumerate(blocks):
    tl = T_LAND0 if bi == 0 else T_LAND; tr = T_RES0 if bi == 0 else T_RES
    for ph, d in (("land", tl), ("resolve", tr), ("settle", T_SET)):
        timeline.append((t, t + d, bi, ph)); t += d
t_stretch = t
timeline.append((t, t + T_OUTRO, NB - 1, "outro")); t += T_OUTRO
DUR = t

k_first = blocks[0]["k"]; k_end = blocks[-1]["k"] + (1 if blocks[-1]["has_rep"] else 0)
t_us0 = steps[k_first - 2]["t_us"] if k_first >= 2 else steps[0]["t_us"]
real_stretch = (steps[k_end - 1]["t_us"] - t_us0) / 1e6
SLOW = t_stretch / real_stretch
stretch_tokens = sum(len(steps[i - 1]["out"]) for i in range(k_first, k_end + 1))
SLOW_TXT = f"{int(round(SLOW))}x"
REAL_TPS = real_tps if real_tps else 0.0

OUTRO_CELLS, OUTRO_ROWS = layout(committed(max(toks) + 1), [])

def frame_state(ts):
    ts = min(ts, DUR - 1e-6)
    for (a, bnd, bi, ph) in timeline:
        if a <= ts < bnd: return bi, ph, ts - a, (ts - a) / (bnd - a)
    return NB - 1, "outro", 0.0, 1.0

def anchor_for(bi, ph, f):
    b = blocks[bi]
    A0 = b["rows_ld"]
    nxt = blocks[bi + 1]["rows_ld"] if bi + 1 < NB else b["rows_s"]
    if ph in ("land", "resolve"): return float(A0)
    if ph == "settle": return A0 + (nxt - A0) * sstep(f)
    return float(nxt)

def draw_cells(d, cells, anchor, bi, ph, u):
    first_row = anchor - NV
    rows_cols = {}
    for (r, c, ch, st, ti) in cells:
        if st not in ("c",): rows_cols.setdefault(r, [10 ** 9, -1]); e = rows_cols[r]; e[0] = min(e[0], c); e[1] = max(e[1], c)
    first_of = {}
    for cc in cells:
        if cc[3] == "draft" and cc[4] not in first_of: first_of[cc[4]] = cc
    for (r, c, ch, st, ti) in sorted(cells, key=lambda q: q[3] == "bonus"):      # bonus chip last so its padding sits on top
        y = (PAD_TOP + (r - first_row) * LH)
        if y < -LH or y > PH: continue
        x = X0 + c * ADV
        bg = None; tx = WHITE; strike = False; pad = 0
        if st == "draft":
            bg = lerp((255, 255, 235), YEL_BG, sstep(u / 0.3)) if ph == "land" else YEL_BG
            tx = YEL_INK
        elif st == "acc":
            if ph == "resolve":
                fl = sstep(u / 0.45); bg = lerp(GRN_HI, GRN_LO, fl); tx = WHITE
            elif ph == "settle": bg = lerp(GRN_LO, BG, sstep(u / T_SET)); tx = WHITE
        elif st == "bonus":
            tx = CY_DK; bg = CY_BG; pad = 7
            if ph == "settle": bg = lerp(CY_BG, BG, sstep(u / T_SET)); tx = lerp(CY_DK, WHITE, sstep(u / T_SET))
        elif st == "rej":
            fade = 1.0
            if ph == "resolve" and u > 0.95: fade = 1 - sstep((u - 0.95) / 0.25)
            if fade <= 0.01: continue
            bg = lerp(BG, RED_BG, fade); tx = lerp(BG, RED_TX, fade); strike = True
        if bg is not None:
            if pad:   # padded cyan chip: wider than one character so the single bonus token reads
                d.rectangle(((x - pad) * SS, (y - 2) * SS, (x + ADV + pad) * SS, (y + LH + 2) * SS), fill=bg,
                            outline=(255, 255, 255) if ph != "settle" else None, width=4)
            else:
                d.rectangle((x * SS, y * SS, (x + ADV) * SS + 1, (y + LH) * SS), fill=bg)
        if st == "draft" and ti is not None and ti > 0 and first_of[ti][0] == r and first_of[ti][1] == c:
            d.rectangle((x * SS, y * SS + 6, x * SS + 6, (y + LH) * SS - 6), fill=(120, 90, 0))     # token boundary tick (wider than v2)
        if ch == "↵":
            d.text(((x + ADV * 0.02) * SS, (y + 8) * SS), "↵", font=FNL, fill=tx)
        else:
            d.text((x * SS, (y + 2) * SS), ch, font=FCODE, fill=tx)
        if strike: d.rectangle((x * SS, (y + LH * 0.5) * SS - 3, (x + ADV) * SS, (y + LH * 0.5) * SS + 2), fill=lerp(BG, (255, 255, 255), fade))
    if ph in ("land", "resolve") and rows_cols:
        col = (lerp((255, 255, 255), (255, 250, 190), sstep(u / 0.3)) if ph == "land" else lerp((255, 250, 190), BG, sstep(u / 0.4)))
        rs = sorted(rows_cols); L = {r: (X0 + rows_cols[r][0] * ADV) * SS for r in rs}; R = {r: (X0 + (rows_cols[r][1] + 1) * ADV) * SS for r in rs}
        top = {r: (PAD_TOP + (r - first_row) * LH) * SS for r in rs}; bot = {r: top[r] + LH * SS for r in rs}
        pts = []
        for r in rs: pts += [(R[r], top[r]), (R[r], bot[r])]
        for r in reversed(rs): pts += [(L[r], bot[r]), (L[r], top[r])]
        d.line(pts + [pts[0]], fill=col, width=8, joint="curve")

FHEAD = font(SANSB, 50); FLEG = font(SANSB, 34); FSPD = font(SANSB, 38)
FBIG = font(MONOB, 76); FMID = font(MONOB, 64); FBLK = font(SANSB, 40); FSEG = font(MONOB, 43); FCHIP = font(SANSB, 50)
FLAB = font(SANSB, 32); FSUBBIG = font(SANSB, 44)
CAP1 = "Clover-35B-A3B on one RTX 5090 through Hermes Agent, self-speculative block drafting (llama.cpp PR build)"
CAP2 = "trace: separate llama-speculative-simple run of the same msm-v2 prompt, not the Hermes session"
def fit(text, path, maxw, hi=34, lo=22):
    for px in range(hi, lo - 1, -1):
        f = font(path, px)
        if f.getlength(text) <= maxw: return f
    return font(path, lo)
FC1 = fit(CAP1, SANSB, 1800); FC2 = fit(CAP2, SANSB, 1800)

# bar geometry: one segment per block, width proportional to drafted + 1 bonus slot (bonus slot drawn a little wider so it reads)
BAR_X, BAR_W, BAR_Y, BAR_H, GAP = 40, 1840, 848, 52, 8
units = [b["n"] + 1 for b in blocks]
SEG_W = [(BAR_W - GAP * (NB - 1)) * u_ / sum(units) for u_ in units]
SEG_X = [BAR_X + sum(SEG_W[:i]) + GAP * i for i in range(NB)]

def render(i):
    ts = i / FPS
    bi, ph, u, f = frame_state(ts)
    b = blocks[bi]
    im = Image.new("RGB", (W, H), BG); dr = ImageDraw.Draw(im)
    # header
    dr.text((40, 12), "writes a whole block at once,", font=FHEAD, fill=WHITE)
    dr.text((40, 76), "then keeps what it believes", font=FHEAD, fill=YEL_TX)
    # right column
    RX = 1240
    lx = [RX, RX + 330]; ly = [26, 82]
    for k_, (lab, bg, tx) in enumerate((("drafted", YEL_SW, YEL_TX), ("kept", GRN_HI, GRN_TX), ("rejected", (235, 60, 70), (255, 130, 140)), ("bonus token", (30, 200, 230), CY_TX))):
        xx, yy = lx[k_ % 2], ly[k_ // 2]
        dr.rounded_rectangle((xx, yy + 4, xx + 40, yy + 44), radius=7, fill=bg)
        dr.text((xx + 54, yy + 2), lab, font=FLEG, fill=tx)
    # code panel (supersampled, camera)
    if ph == "outro":
        cells = OUTRO_CELLS; anchor = float(OUTRO_ROWS); ph_c = "settle"
    else:
        anchor = anchor_for(bi, ph, f)
        cells = {"land": b["draft"], "resolve": b["res"], "settle": b["settle"]}[ph]; ph_c = ph
    layer = Image.new("RGB", (PW * SS, PH * SS), BG); ld = ImageDraw.Draw(layer)
    draw_cells(ld, cells, anchor, bi, ph_c, u if ph != "outro" else 1.0)
    Z0 = 1.0
    if (bi == 0 and ph in ("land", "resolve", "settle")) or (bi == 1 and ph == "land"):
        zf = 1 - sstep((ts - (T_LAND0 + 0.9)) / 1.2)
    else: zf = 0.0
    Z = 1 + (Z0 - 1) * zf
    rows0 = [c[0] for c in blocks[0]["draft"] if c[3] == "draft"]
    first_row0 = blocks[0]["rows_ld"] - NV
    cy0 = PAD_TOP + ((min(rows0) + max(rows0)) / 2 + 0.5 - first_row0) * LH
    ccx = PW / 2; ccy = PH / 2 + (cy0 - PH / 2) * zf
    vw, vh = PW / Z, PH / Z
    bx0 = max(0, min(PW - vw, ccx - vw / 2)); by0 = max(0, min(PH - vh, ccy - vh / 2))
    pnl = layer.resize((PW, PH), Image.LANCZOS, box=(bx0 * SS, by0 * SS, (bx0 + vw) * SS, (by0 + vh) * SS))
    fade = Image.new("L", (PW, 80), 0); fd_ = ImageDraw.Draw(fade)
    for yy in range(80): fd_.line((0, yy, PW, yy), fill=int(255 * (1 - yy / 80) ** 1.3))
    pnl.paste(Image.new("RGB", (PW, 80), BG), (0, 0), fade)
    im.paste(pnl, (PX, PY))
    dr.rectangle((PX - 2, PY - 2, PX + PW + 1, PY + PH + 1), outline=(48, 54, 70), width=2)

    # right column counters
    cy_ = 146
    outro = ph == "outro"
    done = bi if ph == "land" else bi + 1          # blocks whose tally is final (the resolving block counts once it resolves)
    if outro: done = NB
    ra = blocks[done - 1]["run_acc"] if done > 0 else 0
    rd = blocks[done - 1]["run_dr"] if done > 0 else 0
    if outro:
        dr.text((RX, cy_), f"all {NB} blocks", font=FBLK, fill=DIM)
        dr.text((RX, cy_ + 50), f"{ra} of {rd}", font=FBIG, fill=GRN_TX)
        dr.text((RX, cy_ + 146), "drafted tokens kept", font=FSUBBIG, fill=GRN_TX)
        txt = f"+{sum(x['nb'] for x in blocks)} bonus tokens"
        cw = FCHIP.getlength(txt) + 56; cyc = cy_ + 214
        dr.rounded_rectangle((RX, cyc, RX + cw, cyc + 74), radius=14, fill=CY_BG); dr.text((RX + 28, cyc + 10), txt, font=FCHIP, fill=CY_DK)
        dr.text((RX, 470), "summed from the per-block", font=FLAB, fill=DIM)
        dr.text((RX, 506), "tallies in the bar", font=FLAB, fill=DIM)
    else:
        dr.text((RX, cy_), f"block {bi + 1} of {NB}", font=FBLK, fill=DIM)
        if ph == "land":
            dr.text((RX, cy_ + 50), f"drafted {b['n']}", font=FBIG, fill=YEL_TX)
            dr.text((RX, cy_ + 146), "all at once", font=FSUBBIG, fill=YEL_TX)
            chip_on = False
        else:
            dr.text((RX, cy_ + 50), f"kept {b['acc']} of {b['n']}", font=FBIG, fill=GRN_TX)
            dr.text((RX, cy_ + 146), f"{b['nrej']} rejected" if b["nrej"] else "none rejected", font=FSUBBIG, fill=(255, 130, 140))
            chip_on = True
        cyc = cy_ + 214
        txt = f"+{b['nb']} bonus token"
        cw = FCHIP.getlength(txt) + 56
        if chip_on: dr.rounded_rectangle((RX, cyc, RX + cw, cyc + 74), radius=14, fill=CY_BG); dr.text((RX + 28, cyc + 10), txt, font=FCHIP, fill=CY_DK)
        else: dr.rounded_rectangle((RX, cyc, RX + cw, cyc + 74), radius=14, outline=(120, 132, 156), width=3); dr.text((RX + 28, cyc + 10), txt, font=FCHIP, fill=(120, 132, 156))
        if done > 0:
            ky = 470
            dr.text((RX, ky), "kept so far", font=FLAB, fill=DIM)
            dr.text((RX, ky + 36), f"{ra} of {rd}", font=FMID, fill=GRN_TX)
            dr.text((RX, ky + 112), "of drafted tokens, blocks so far", font=FLAB, fill=DIM)
    dr.text((RX, 660), f"real speed {REAL_TPS:.0f} tok/s", font=FSPD, fill=(255, 255, 255))
    dr.text((RX, 708), f"shown slowed ~{SLOW_TXT}", font=FSPD, fill=(255, 255, 255))
    if outro:
        dr.text((RX, 764), f"trace: {N} verify steps, {real_tok} tokens", font=FLAB, fill=DIM)
        dr.text((RX, 802), f"shown: steps {FIRST} to {LAST}", font=FLAB, fill=DIM)

    # block bar, fills left to right
    for j in range(NB):
        x0 = SEG_X[j]; w = SEG_W[j]; bj = blocks[j]
        unit = w / (bj["n"] + 1.6)          # the bonus slot is drawn 1.6 units wide so it reads
        if j > done or (j == done and not (j == bi and ph in ("land", "resolve") and not outro)):
            dr.rounded_rectangle((x0, BAR_Y, x0 + w, BAR_Y + BAR_H), radius=8, outline=(56, 62, 80), width=3)
            continue
        if j == bi and ph == "land" and not outro:
            dr.rounded_rectangle((x0, BAR_Y, x0 + w, BAR_Y + BAR_H), radius=8, fill=YEL_LN)
            continue
        # resolved (or resolving) segment
        res_u = u if (j == bi and ph == "resolve" and not outro) else 9
        ka = max(bj["acc"] * unit, 14) if bj["acc"] else 0       # slivers get a 14 px floor so they read
        gcol = lerp(GRN_HI, (36, 160, 78), sstep(res_u / 0.6)); rcol = lerp((235, 60, 70), (190, 44, 58), sstep((res_u - 0.5) / 0.5))
        if res_u < 9: yc = lerp(YEL_LN, gcol, sstep(res_u / 0.35)); rc2 = lerp(YEL_LN, rcol, sstep(res_u / 0.35)); bc = lerp(YEL_LN, CY_BG, sstep(res_u / 0.35))
        else: yc = gcol; rc2 = rcol; bc = CY_BG
        dr.rounded_rectangle((x0, BAR_Y, x0 + ka - 2, BAR_Y + BAR_H), radius=6, fill=yc) if ka > 4 else None
        rw = max(bj["nrej"] * unit, 14) if bj["nrej"] else 0
        if rw > 2: dr.rounded_rectangle((x0 + ka, BAR_Y, x0 + ka + rw - 2, BAR_Y + BAR_H), radius=6, fill=rc2)
        dr.rounded_rectangle((x0 + ka + rw, BAR_Y, x0 + w, BAR_Y + BAR_H), radius=6, fill=bc)
        lab = f"{bj['acc']}/{bj['n']}"
        dr.text((x0 + w / 2 - FSEG.getlength(lab) / 2, BAR_Y + BAR_H + 4), lab, font=FSEG, fill=WHITE if (j < done or outro) else DIM)
    dr.text((BAR_X, BAR_Y - 42), "each block, in order: tokens kept / tokens drafted", font=FLAB, fill=DIM)
    # caption
    dr.text((40, 972), CAP1, font=FC1, fill=(222, 226, 236))
    dr.text((40, 1020), CAP2, font=FC2, fill=(200, 206, 220))
    return im

def to_bytes(i): return render(i).tobytes()

if __name__ == "__main__":
    total = int(math.ceil(DUR * FPS))
    print(f"blocks {NB} steps {FIRST}-{LAST} dur {DUR:.1f}s frames {total} real_stretch {real_stretch*1000:.0f} ms slow {SLOW:.1f}x "
          f"stretch tokens {stretch_tokens} stretch kept {blocks[-1]['run_acc']} of {blocks[-1]['run_dr']} real {REAL_TPS} cols {COLS} rows {NV} cap fonts {FC1.size},{FC2.size}", flush=True)
    if PNG_DIR:
        os.makedirs(PNG_DIR, exist_ok=True)
        for tt in PNG_AT: render(int(tt * FPS)).save(f"{PNG_DIR}/t{tt:05.2f}.png")
        sys.exit(0)
    cmd = [FF, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-framerate", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "medium", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(POOL) as pool:
        for b_ in pool.imap(to_bytes, range(total), chunksize=4): p.stdin.write(b_)
    p.stdin.close(); rc = p.wait()
    print("ffmpeg rc", rc, "size", os.path.getsize(out) if os.path.exists(out) else None)
    sys.exit(rc)
