"""Build a behavior-cloning dataset of optimal maze trajectories and train a TinyGPT from
scratch. `lam` is the world-model loss weight: lam=0 = action-only arm, lam>0 adds the free
cross-entropy on observation tokens. Same data/model/forward across arms — only the mask
weights differ. Trajectories are optimal paths only (no recovery demos) on purpose: that is
where anticipating the maze (a world model) can pay off. Torch; runs on capsule."""
import random
import torch
import torch.nn.functional as F

from lib.echo_maze.maze import gen_maze, random_endpoints
from lib.echo_maze.encode import encode_trajectory, build_loss_mask, PAD, VOCAB_SIZE
from lib.echo_maze.model import TinyGPT, GPTConfig


def build_dataset(sizes, n_per_size, seed, block_size):
    """List of (tokens, roles). Mazes/endpoints drawn from one seeded RNG; over-length dropped."""
    rng = random.Random(seed)
    data = []
    for size in sizes:
        made = 0
        while made < n_per_size:
            maze = gen_maze(size, rng)
            start, goal = random_endpoints(size, rng)
            tokens, roles = encode_trajectory(maze, start, goal)
            if len(tokens) > block_size:
                continue
            data.append((tokens, roles))
            made += 1
    return data


def _pad(seq, n, val):
    return seq + [val] * (n - len(seq))


def train(lam, seed, sizes, n_train_per_size=4000, steps=3000, batch_size=64,
          lr=3e-4, device="cuda", block_size=512):
    torch.manual_seed(seed)
    random.seed(seed)
    data = build_dataset(sizes, n_train_per_size, seed, block_size)
    idx_all = torch.tensor([_pad(t, block_size, PAD) for t, _ in data], dtype=torch.long)
    w_all = torch.tensor([_pad(build_loss_mask(r, lam), block_size, 0.0) for _, r in data],
                         dtype=torch.float)
    N = idx_all.shape[0]

    model = TinyGPT(GPTConfig(block_size=block_size)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    g = torch.Generator().manual_seed(seed)
    log = []
    model.train()
    for step in range(steps):
        bi = torch.randint(0, N, (batch_size,), generator=g)
        idx = idx_all[bi].to(device)
        w = w_all[bi].to(device)
        logits = model(idx)                              # [B,T,V]
        lt = logits[:, :-1, :].reshape(-1, VOCAB_SIZE)   # predict token t+1 from position t
        tgt = idx[:, 1:].reshape(-1)
        wt = w[:, 1:].reshape(-1)                        # weight aligned to the TARGET token
        ce = F.cross_entropy(lt, tgt, reduction="none")
        loss = (ce * wt).sum() / wt.sum().clamp_min(1.0)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if step % 200 == 0 or step == steps - 1:
            log.append({"step": step, "loss": float(loss.detach())})
    return model, log
