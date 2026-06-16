"""ECHO maze microcosm — torch units. Run on capsule: ~/unsloth-env/bin/python -m pytest …"""
import torch
from lib.echo_maze.model import TinyGPT, GPTConfig
from lib.echo_maze.encode import VOCAB_SIZE


def test_param_count_near_10m():
    n = TinyGPT(GPTConfig()).num_params()
    assert 8_000_000 <= n <= 12_000_000, n


def test_forward_shape():
    m = TinyGPT(GPTConfig(block_size=64))
    idx = torch.randint(0, VOCAB_SIZE, (2, 16))
    out = m(idx)
    assert out.shape == (2, 16, VOCAB_SIZE)


def test_overfit_one_batch():
    torch.manual_seed(0)
    m = TinyGPT(GPTConfig(block_size=32))
    idx = torch.randint(0, VOCAB_SIZE, (4, 32))
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    import torch.nn.functional as F
    first = last = None
    for step in range(60):
        logits = m(idx)
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, VOCAB_SIZE), idx[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 0:
            first = float(loss)
        last = float(loss)
    assert last < first * 0.5   # memorizes a fixed batch
