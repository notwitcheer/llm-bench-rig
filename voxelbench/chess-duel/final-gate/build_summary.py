#!/usr/bin/env python3
"""build_summary.py <root> --a-name A --b-name B [--a-regime T --b-regime T] [--movetime 1000]

Reads what the CPU phase produced under <root> (gate-a/gate.json, gate-b/gate.json, match/match.json, anchor-a/anchor.json, anchor-b/anchor.json, agent-a/ agent-b/ run
records) and writes <root>/summary.json (everything), <root>/scoreboard.json (the renderer's input), <root>/line1.txt (the PRE-REGISTERED line-1 variant, brackets filled
from the data by rule, nothing worded by hand) and <root>/result.md (the table the readout cron pastes).

LINE-1 RULE (pre-registered in the 81 item header, 2026-11-03):
  V1 both pass the gate   -> "<A> and <B> each wrote a UCI chess engine in one Hermes /goal hour on one RTX 5090. Both pass perft 1-5 and play legal chess. <result sentence>"
  V2 exactly one fails    -> "<P> passes perft 1-5 and plays legal chess; <F> fails the gate: <failure line>. <result sentence if the match was played>"
  V3 both fail            -> "Neither engine passes the gate: <A>: <failure line>. <B>: <failure line>. <result sentence if the match was played>"
  result sentence (match of 20 games): margin = |points difference|: >= 3 "<W> won the 20-game match x-y", 1 to 2.5 "<W> edged the 20-game match x-y (inside the noise of 20 games)",
  0 "the 20-game match ended level x-y"; match not played (an engine never answered `uci`) "no match: <name> never answered `uci`".
  coverage add-on (passing engines only): " <name> passes the gate but fails the <position> coverage extra (<covers>)."  (castling / en passant / promotion bugs the start position cannot show)
"""
import glob
import json
import os
import sys


def arg(name, default=None):
    return sys.argv[sys.argv.index(name) + 1] if name in sys.argv else default


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def fail_line(g):
    if g is None:
        return "no gate result (no engine.py)"
    if g.get("no_engine"):
        return "no engine.py was written"
    f = (g.get("failed") or ["?"])[0]
    d = (g.get("detail") or {}).get(f, "")
    return "%s: %s" % (f, d)


def agent_row(root, k):
    d = load(os.path.join(root, "agent-" + k, "driver.json")) or {}
    m = load(os.path.join(root, "agent-" + k, "metrics.json")) or {}
    return d, m


def main():
    root = sys.argv[1]
    A, B = arg("--a-name", "A"), arg("--b-name", "B")
    mt = int(arg("--movetime", 1000))
    names = {"a": A, "b": B}
    S = {"names": names, "engines": {}, "movetime_ms": mt}
    gates = {k: load(os.path.join(root, "gate-" + k, "gate.json")) for k in "ab"}
    match = load(os.path.join(root, "match", "match.json"))
    anchors = {k: load(os.path.join(root, "anchor-" + k, "anchor.json")) for k in "ab"}
    for k in "ab":
        g = gates[k]
        passed = bool(g and g.get("pass"))
        d, m = agent_row(root, k)
        code = ""
        ep = glob.glob(os.path.join(root, "agent-" + k, "work", "engine.py"))
        if ep:
            src = open(ep[0], encoding="utf-8", errors="replace").read()
            code = "%d lines, %.1f KB" % (src.count("\n") + 1, len(src.encode()) / 1000.0)
        rows = [("code", code or "-"), ("agent wall", ("%d min (%s)" % (round(d.get("wall_s", 0) / 60), d.get("end"))) if d else "-")]
        if m:
            rows += [("api calls / tool calls", "%s / %s" % (m.get("api_calls", "-"), m.get("tool_calls", "-")))]
        if g:
            pf = g.get("perft", {})
            rows += [("perft 1-5", " ".join("ok" if (pf.get(str(i)) or {}).get("got") == (pf.get(str(i)) or {}).get("want") else "X" for i in range(1, 6))),
                     ("perft depth 5", ("%s s" % pf["5"]["secs"]) if pf.get("5") and pf["5"].get("got") is not None else "-")]
            gm = g.get("games") or {}
            rows += [("10 games vs baseline", "%s-%s-%s, illegal %s, time/crash %s" % (gm.get("engine_w", "-"), gm.get("draw", "-"), gm.get("engine_l", "-"), gm.get("illegal", "-"), gm.get("time_or_crash", "-")))]
            cov = g.get("coverage_failures") or []
            rows += [("coverage extras", "all ok" if not cov and g.get("extras") else ("gaps: " + ", ".join(cov) if cov else "-"))]
        an = anchors[k]
        if an and an.get("anchor"):
            x = an["anchor"]
            rows += [("points vs Stockfish (limited to 1320)", "%s/10%s" % (("%g" % x["score"]), " (bound)" if x["bounded"] else ""))]
        S["engines"][k] = {"name": names[k], "regime": arg("--%s-regime" % k, ""), "gate_pass": passed, "gate_line": ("perft 1-5 exact, legal in all games" if passed else fail_line(g)),
                           "rows": rows, "coverage_failures": (g or {}).get("coverage_failures", []), "gate": g and {"pass": g.get("pass"), "failed": g.get("failed")}}
    # match sentence
    if match:
        sa, sb = match["score_a"], match["score_b"]
        diff = abs(sa - sb)
        w, l = (A, B) if sa > sb else (B, A)
        hi, lo = max(sa, sb), min(sa, sb)
        if diff >= 3:
            msent = "%s won the 20-game match %g-%g." % (w, hi, lo)
        elif diff >= 1:
            msent = "%s edged the 20-game match %g-%g (inside the noise of 20 games)." % (w, hi, lo)
        else:
            msent = "the 20-game match ended level %g-%g." % (sa, sb)
        t = match["tally"]
        extra = []
        if t["a_illegal"] or t["b_illegal"]:
            extra.append("illegal moves: %s %d, %s %d" % (A, t["a_illegal"], B, t["b_illegal"]))
        if t["a_time_or_crash"] or t["b_time_or_crash"]:
            extra.append("forfeits (late move or crash): %s %d, %s %d" % (A, t["a_time_or_crash"], B, t["b_time_or_crash"]))
        S["match"] = {"played": True, "a": sa, "b": sb, "tally": t, "sentence": msent, "extra": extra}
    else:
        who = [names[k] for k in "ab" if (gates[k] or {}).get("checks", {}).get("uci_handshake") is not True]
        msent = "no match: %s never answered `uci`." % (" / ".join(who) if who else "an engine")
        S["match"] = {"played": False, "sentence": msent, "extra": []}
    pa, pb = S["engines"]["a"]["gate_pass"], S["engines"]["b"]["gate_pass"]
    if pa and pb:
        v, l1 = "V1", "%s and %s each wrote a UCI chess engine in one Hermes /goal hour on one RTX 5090. Both pass perft 1-5 and play legal chess. %s" % (A, B, S["match"]["sentence"][0].upper() + S["match"]["sentence"][1:])
    elif pa or pb:
        p, f = ("a", "b") if pa else ("b", "a")
        v, l1 = "V2", "%s passes perft 1-5 and plays legal chess; %s fails the gate: %s. %s" % (names[p], names[f], S["engines"][f]["gate_line"], S["match"]["sentence"][0].upper() + S["match"]["sentence"][1:])
    else:
        v, l1 = "V3", "Neither engine passes the gate: %s: %s. %s: %s. %s" % (A, S["engines"]["a"]["gate_line"], B, S["engines"]["b"]["gate_line"], S["match"]["sentence"][0].upper() + S["match"]["sentence"][1:])
    for k in "ab":
        g = gates[k]
        if g and g.get("pass"):
            for c in g.get("coverage_failures", []):
                covers = (g.get("extras", {}).get(c) or {}).get("covers", "")
                l1 += " %s passes the gate but fails the %s coverage extra (%s)." % (names[k], c, covers)
    S["variant"], S["line1"] = v, l1
    # scoreboard json for the renderer
    if match:
        sc = "%g - %g" % (match["score_a"], match["score_b"])
        head = "20 games, %d ms per move, alternating colours: %s  vs  %s" % (mt, A, B)
    else:
        sc, head = "no match", "an engine never answered `uci`"
    SB = {"engines": S["engines"], "sub": "same prompt, one /goal hour each, then a referee they never saw", "match": {"headline": head, "score": sc, "note": " | ".join(S["match"]["extra"]) or S["match"]["sentence"]},
          "footer": "gate = UCI handshake, perft 1-5 from the start position, 10 games vs a random mover with no illegal move (more perft positions are reported, not gating). match = 20 games, illegal move = loss. Stockfish anchor: 10 games each, indicative only."}
    json.dump(S, open(os.path.join(root, "summary.json"), "w"), indent=1)
    json.dump(SB, open(os.path.join(root, "scoreboard.json"), "w"), indent=1)
    open(os.path.join(root, "line1.txt"), "w").write(l1 + "\n")
    md = ["| | %s | %s |" % (A, B), "|---|---|---|", "| gate | %s | %s |" % ("PASS" if pa else "FAIL", "PASS" if pb else "FAIL"), "| gate line | %s | %s |" % (S["engines"]["a"]["gate_line"], S["engines"]["b"]["gate_line"])]
    ra, rb = dict(S["engines"]["a"]["rows"]), dict(S["engines"]["b"]["rows"])
    for key in dict.fromkeys(list(ra) + list(rb)):
        md.append("| %s | %s | %s |" % (key, ra.get(key, "-"), rb.get(key, "-")))
    md += ["", "variant %s" % v, "", "line 1: " + l1]
    open(os.path.join(root, "result.md"), "w").write("\n".join(md) + "\n")
    print(v, "|", l1)


if __name__ == "__main__":
    main()
