#!/usr/bin/env python3
"""c94_count.py: the llama-server token count of the HUD, as a pure function (imported by c94_proxy.py live and by c94_recount.py offline, so the proof runs the production formula)."""


def llama_srv_tokens(counters, slot_n_decoded=None):
    """server-side token count for a llama-server upstream, from the proxy's own per-agent counters.
    finished requests = exact predicted_n (or, for a stream cut short, its deltas); the request in flight = the slot's own n_decoded when it agrees with the deltas
    seen so far (3% + 32 tokens: llama.cpp sends one delta per token but a few tokens, e.g. template markers, have none), else the deltas. Single agent only for n_decoded.
    A request whose `timings` have arrived (final_seen) is never read from the slot: the slot keeps reporting its full n_decoded for a moment after the last chunk."""
    tot = 0
    for v in counters.values():
        tot += v["srv_done"]
        if v["inflight"] > 0 and not v["final_seen"]:
            cur = v["chunks"] - v["done_chunks"]
            if slot_n_decoded is not None and len(counters) == 1 and abs(slot_n_decoded - cur) <= 0.03 * cur + 32:
                cur = slot_n_decoded
            tot += cur
    return tot


def basalt_srv_tokens(counters, slot_generated=None):
    """Basalt server-side token count: exact predicted_n of finished requests (timings of the last chunk) + the decoding slot's own `generated` for the request in flight.
    Basalt streams text pieces per verify window (up to 8 accepted tokens), so SSE deltas under-count and are never used here. A request whose timings arrived is never read from the slot."""
    tot = 0
    for v in counters.values():
        tot += v["srv_done"]
        if v["inflight"] > 0 and not v["final_seen"] and slot_generated is not None and len(counters) == 1:
            tot += slot_generated
    return tot
