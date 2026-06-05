from lib.speed_vllm import prefill_decode_tps


def test_prefill_decode_tps_from_timings():
    # 512 prompt tokens over 100ms TTFT -> 5120 tok/s prefill;
    # 128 output tokens over the remaining 1.0s -> 128 tok/s decode.
    r = prefill_decode_tps(prompt_tokens=512, ttft_s=0.1, output_tokens=128, total_s=1.1)
    assert round(r["prefill_tps"]) == 5120
    assert round(r["decode_tps"]) == 128
