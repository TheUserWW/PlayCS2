"""Causal Transformer blocks for the CS2 policy timeline."""

import torch
from torch import nn
from torch.nn import functional as F

from .config import CS2PolicyConfig


class CausalSelfAttention(nn.Module):
    """Fused multi-head Q/K/V; masks use True for allowed attention."""

    def __init__(self, config: CS2PolicyConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.n_embed // config.n_head
        self.qkv = nn.Linear(config.n_embed, 3 * config.n_embed, bias=False)
        self.projection = nn.Linear(config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x, attention_mask=None):
        b, t, width = x.shape
        q, k, v = self.qkv(x).reshape(b, t, 3, self.n_head, self.head_size).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=attention_mask is None,
        )
        x = x.transpose(1, 2).contiguous().reshape(b, t, width)
        return self.dropout(self.projection(x))


class Block(nn.Module):
    def __init__(self, config: CS2PolicyConfig):
        super().__init__()
        self.attention = CausalSelfAttention(config)
        self.feed_forward = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed), nn.ReLU(),
            nn.Linear(4 * config.n_embed, config.n_embed), nn.Dropout(config.dropout),
        )
        self.attention_norm = nn.LayerNorm(config.n_embed)
        self.feed_forward_norm = nn.LayerNorm(config.n_embed)

    def forward(self, x, attention_mask=None):
        x = x + self.attention(self.attention_norm(x), attention_mask)
        return x + self.feed_forward(self.feed_forward_norm(x))
