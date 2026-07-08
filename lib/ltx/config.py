"""LTX-2 two-stage pipeline config invariants (tested — ADR-0003).

The vendor DistilledPipeline hard-requires H and W divisible by 64 (two-stage:
stage 1 runs at half resolution) and a frame count of the form 8k+1. A bad
matrix entry must fail on the Mac at config time, not mid-run on the GPU.
"""


def validate_video_config(width: int, height: int, num_frames: int) -> None:
    """Raise ValueError unless (width, height, num_frames) is pipeline-legal."""
    if width % 64 != 0 or height % 64 != 0:
        raise ValueError(
            f"resolution {width}x{height} is not divisible by 64 "
            "(two-stage DistilledPipeline requirement)"
        )
    if num_frames < 1 or (num_frames - 1) % 8 != 0:
        raise ValueError(
            f"num_frames {num_frames} is not of the form 8k+1 "
            "(LTX-2 latent temporal stride)"
        )
