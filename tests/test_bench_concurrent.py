from scripts.bench_concurrent import aggregate_tps


def test_aggregate_tps():
    # 512 output tokens over 2.0s wall -> 256 tok/s aggregate
    assert aggregate_tps(total_output_tokens=512, wall_s=2.0) == 256.0


def test_aggregate_tps_zero_wall():
    assert aggregate_tps(total_output_tokens=10, wall_s=0) == 0.0
