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


from lib.echo_maze.train import train, build_dataset


def test_build_dataset_shapes_and_padding():
    data = build_dataset(sizes=[5], n_per_size=8, seed=0, block_size=128)
    assert len(data) == 8
    tokens, roles = data[0]
    assert len(tokens) == len(roles) and roles[0] == "bos" and roles[-1] == "eos"


def test_train_smoke_loss_drops():
    _, log = train(lam=1.0, seed=0, sizes=[5], n_train_per_size=64, steps=80,
                    batch_size=16, device="cpu", block_size=128)
    assert log[-1]["loss"] < log[0]["loss"]
