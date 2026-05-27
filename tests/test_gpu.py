import pytest
from lib.gpu import parse_nvidia_smi, get_gpu_stats

SAMPLE_SMI_OUTPUT = """name, memory.used [MiB], memory.total [MiB], utilization.gpu [%], temperature.gpu
NVIDIA GeForce RTX 5090, 4523 MiB, 32107 MiB, 87 %, 62"""

def test_parse_nvidia_smi():
    stats = parse_nvidia_smi(SAMPLE_SMI_OUTPUT)
    assert stats["name"] == "NVIDIA GeForce RTX 5090"
    assert stats["vram_used_mib"] == 4523
    assert stats["vram_total_mib"] == 32107
    assert stats["gpu_util_pct"] == 87
    assert stats["temp_c"] == 62

def test_parse_nvidia_smi_no_spaces():
    raw = """name, memory.used [MiB], memory.total [MiB], utilization.gpu [%], temperature.gpu
NVIDIA GeForce RTX 5090, 0 MiB, 32107 MiB, 0 %, 35"""
    stats = parse_nvidia_smi(raw)
    assert stats["vram_used_mib"] == 0
    assert stats["gpu_util_pct"] == 0
