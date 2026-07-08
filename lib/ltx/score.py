"""Pure aggregation of ltx_synth records (tested — no GPU, no I/O).

Groups records by resolution config and computes the headline economics:
mean generation seconds, seconds of compute per second of output video
(the honest inverse-RTF for a fixed-length clip), and true peak VRAM.
"""


def summarize(records: list) -> dict:
    configs = {}
    for r in records:
        key = f"{r['width']}x{r['height']}"
        configs.setdefault(key, []).append(r)

    out = {}
    for key, rows in configs.items():
        ok = [r for r in rows if r.get("ok")]
        failed = [r for r in rows if not r.get("ok")]
        if ok:
            mean_gen = sum(r["gen_seconds"] for r in ok) / len(ok)
            video_s = ok[0]["num_frames"] / ok[0]["fps"]
            spvs = mean_gen / video_s
            max_peak = max(r["peak_vram_mib"] for r in ok)
        else:
            mean_gen = video_s = spvs = max_peak = None
        out[key] = {
            "n_ok": len(ok),
            "n_failed": len(failed),
            "mean_gen_seconds": mean_gen,
            "video_seconds": video_s,
            "seconds_per_video_second": spvs,
            "max_peak_vram_mib": max_peak,
        }
    return {"configs": out}
