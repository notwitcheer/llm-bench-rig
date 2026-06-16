"""A small from-scratch decoder-only transformer (nanoGPT-style) for the maze microcosm.
~10.6M params at the default config. Torch only; runs on capsule (sm_120)."""
from dataclasses import dataclass
import torch
import torch.nn as nn

from lib.echo_maze.encode import VOCAB_SIZE


@dataclass
class GPTConfig:
    vocab_size: int = VOCAB_SIZE
    block_size: int = 512
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = nn.MultiheadAttention(cfg.n_embd, cfg.n_head, dropout=cfg.dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd), nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd), nn.Dropout(cfg.dropout),
        )

    def forward(self, x, attn_mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=attn_mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or GPTConfig()
        self.tok_emb = nn.Embedding(self.cfg.vocab_size, self.cfg.n_embd)
        self.pos_emb = nn.Embedding(self.cfg.block_size, self.cfg.n_embd)
        self.drop = nn.Dropout(self.cfg.dropout)
        self.blocks = nn.ModuleList([Block(self.cfg) for _ in range(self.cfg.n_layer)])
        self.ln_f = nn.LayerNorm(self.cfg.n_embd)
        self.head = nn.Linear(self.cfg.n_embd, self.cfg.vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos)[None, :, :])
        mask = torch.triu(torch.ones(T, T, device=idx.device, dtype=torch.bool), diagonal=1)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
