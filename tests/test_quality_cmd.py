from lib.quality import build_server_command


def test_server_command_default_full_vram():
    cmd = build_server_command("/m/model.gguf", port=8090, ngl=99)
    assert cmd[cmd.index("--n-gpu-layers") + 1] == "99"
    assert "--n-cpu-moe" not in cmd
    assert cmd[cmd.index("--port") + 1] == "8090"


def test_server_command_with_ctx_size():
    cmd = build_server_command("/m/model.gguf", port=8090, ngl=99, ctx_size=65536)
    assert cmd[cmd.index("--ctx-size") + 1] == "65536"


def test_server_command_with_cpu_moe():
    cmd = build_server_command("/m/model.gguf", port=8090, ngl=99, n_cpu_moe=28)
    assert cmd[cmd.index("--n-cpu-moe") + 1] == "28"
