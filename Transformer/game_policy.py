"""CNN + action/state timeline + the project's causal Transformer blocks.

This is a policy architecture, not a trainer or a pretrained CS2 player. No text
tokenizer, game process, screen capture or keyboard APIs are imported here.
"""

import torch
from torch import nn
from torch.nn import functional as F

from cs2.actions import ACTION_DIM, BUTTONS, BRANCHES, YAW_BINS, PITCH_BINS
from cs2.state import STATE_DIM
from .config import CS2PolicyConfig
from .model import Block


class FrameEncoder(nn.Module):
    """Encode individual RGB frames, retaining a coarse 3x5 spatial grid."""

    def __init__(self, cfg):
        super().__init__()
        self.size = (cfg.image_height, cfg.image_width)
        self.mask_rects = cfg.mask_rects
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.GroupNorm(8, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(8, 64), nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1), nn.GroupNorm(8, 96), nn.GELU(),
            nn.Conv2d(96, 128, 3, stride=2, padding=1), nn.GroupNorm(8, 128), nn.GELU(),
            nn.AdaptiveAvgPool2d((3, 5)), nn.Flatten(),
            nn.Linear(128 * 3 * 5, cfg.vision_dim), nn.LayerNorm(cfg.vision_dim),
        )

    def forward(self, images):
        if images.ndim != 5 or images.shape[2] != 3 or min(images.shape[-2:]) < 1:
            raise ValueError("images must be [batch, time, 3, height, width] RGB")
        if images.dtype != torch.uint8:
            raise ValueError("images must be uint8 RGB in [0,255]; normalization is internal")
        b, t, c, h, w = images.shape
        x = images.reshape(b * t, c, h, w).float() / 255.0
        x = F.interpolate(x, size=self.size, mode="bilinear", align_corners=False)
        for x0, y0, x1, y1 in self.mask_rects:
            x[:, :, int(y0*self.size[0]):int(y1*self.size[0]), int(x0*self.size[1]):int(x1*self.size[1])] = 0
        return self.cnn(x).reshape(b, t, -1)


class CS2Policy(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = cfg = config or CS2PolicyConfig()
        self.register_buffer('action_availability', torch.ones(len(BRANCHES), dtype=torch.bool))
        self.frame_encoder = FrameEncoder(cfg)
        self.vision_projection = nn.Linear(cfg.vision_dim, cfg.n_embed)
        self.action_encoder = nn.Sequential(nn.Linear(ACTION_DIM, cfg.n_embed), nn.GELU())
        # Value + per-feature availability, plus GSI age and freshness.
        self.state_encoder = nn.Sequential(nn.Linear(STATE_DIM * 2 + 2, cfg.n_embed), nn.GELU())
        self.time_encoder = nn.Sequential(nn.Linear(2, cfg.n_embed), nn.GELU(), nn.Linear(cfg.n_embed, cfg.n_embed))
        self.position_embedding = nn.Embedding(cfg.block_size, cfg.n_embed)
        self.input_norm = nn.LayerNorm(cfg.n_embed)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.final_norm = nn.LayerNorm(cfg.n_embed)
        self.heads = nn.ModuleDict({
            "forward_logits": nn.Linear(cfg.n_embed, 3),
            "strafe_logits": nn.Linear(cfg.n_embed, 3),
            "button_logits": nn.Linear(cfg.n_embed, len(BUTTONS)),
            "weapon_logits": nn.Linear(cfg.n_embed, 6),
            "yaw_logits": nn.Linear(cfg.n_embed, len(YAW_BINS)),
            "pitch_logits": nn.Linear(cfg.n_embed, len(PITCH_BINS)),
            "angle_residual": nn.Linear(cfg.n_embed, 2),
            "value": nn.Linear(cfg.n_embed, 1),
        })

    def encode_frames(self, images):
        return self.frame_encoder(images)

    def forward(self, images, previous_actions, state_values, state_available,
                gsi_age, gsi_fresh, timestamps, *, valid=None, episode_starts=None):
        return self.forward_features(
            self.encode_frames(images), previous_actions, state_values, state_available,
            gsi_age, gsi_fresh, timestamps, valid=valid, episode_starts=episode_starts)

    def forward_features(self, visual_features, previous_actions, state_values, state_available,
                         gsi_age, gsi_fresh, timestamps, *, valid=None, episode_starts=None):
        """All sequences are [B,T,...]; timestamps are float64 monotonic seconds.

        Each action input belongs to the PRECEDING interval, never the action being
        predicted. `valid` supports right padding. `episode_starts` isolates packed
        episodes (including deaths/round resets); first actions there are ignored.
        """
        if visual_features.ndim != 3:
            raise ValueError("visual_features must be [B,T,vision_dim]")
        b, t, d = visual_features.shape
        if b < 1 or not 1 <= t <= self.config.block_size or d != self.config.vision_dim:
            raise ValueError("Invalid visual feature shape or context exceeds block_size")
        expected = {"previous_actions": (previous_actions, (b, t, ACTION_DIM)),
                    "state_values": (state_values, (b, t, STATE_DIM)),
                    "state_available": (state_available, (b, t, STATE_DIM)),
                    "gsi_age": (gsi_age, (b, t)), "gsi_fresh": (gsi_fresh, (b, t)),
                    "timestamps": (timestamps, (b, t))}
        for name, (tensor, shape) in expected.items():
            if tensor.shape != shape or tensor.device != visual_features.device:
                raise ValueError(f"{name} must have shape {shape} on the visual features device")
            if not torch.isfinite(tensor).all():
                raise ValueError(f"{name} must be finite")
        if not torch.isfinite(visual_features).all():
            raise ValueError("visual_features must be finite")
        device = visual_features.device
        if valid is None:
            valid = torch.ones((b, t), dtype=torch.bool, device=device)
        if episode_starts is None:
            episode_starts = torch.zeros((b, t), dtype=torch.bool, device=device)
        for name, mask in (("valid", valid), ("episode_starts", episode_starts)):
            if mask.shape != (b, t) or mask.dtype != torch.bool or mask.device != device:
                raise ValueError(f"{name} must be boolean [B,T] on the model input device")
        if not valid[:, 0].all() or (valid[:, 1:] & ~valid[:, :-1]).any():
            raise ValueError("Each sequence must contain data followed by optional right padding")
        if state_available.dtype != torch.bool or gsi_fresh.dtype != torch.bool:
            raise ValueError("state_available and gsi_fresh must be boolean")
        if (gsi_age < 0).any():
            raise ValueError("gsi_age cannot be negative")
        starts = episode_starts.clone()
        starts[:, 0] = True  # The context window starts a new position/time origin.
        indexes = torch.arange(t, device=device).expand(b, -1)
        origins = torch.where(starts, indexes, 0).cummax(dim=1).values
        positions = indexes - origins
        segments = starts.long().cumsum(dim=1)
        causal = indexes[:, :, None] >= indexes[:, None, :]
        allowed = causal & (segments[:, :, None] == segments[:, None, :]) & valid[:, None, :]
        # Padding queries attend only themselves; never create an all-masked row.
        allowed = torch.where(valid[:, :, None], allowed, torch.eye(t, dtype=torch.bool, device=device)[None])
        allowed = allowed[:, None, :, :]
        delta = torch.zeros_like(timestamps)
        delta[:, 1:] = timestamps[:, 1:] - timestamps[:, :-1]
        if ((delta < 0) & valid & ~starts).any():
            raise ValueError("timestamps must be nondecreasing within each episode")
        delta = torch.where(starts | ~valid, 0.0, delta)
        elapsed = timestamps - timestamps.gather(1, origins)
        elapsed = torch.where(valid, elapsed, 0.0)
        dtype = visual_features.dtype
        time_features = torch.stack((torch.log1p(delta.clamp(min=0)), torch.log1p(elapsed.clamp(min=0))), dim=-1).to(dtype)
        actions = previous_actions.masked_fill(episode_starts[..., None], 0).to(dtype)
        available = state_available & gsi_fresh[..., None]
        state = torch.cat((state_values.masked_fill(~available, 0).to(dtype), available.to(dtype),
                           torch.log1p(gsi_age.clamp(max=60))[..., None].to(dtype),
                           gsi_fresh[..., None].to(dtype)), dim=-1)
        x = self.input_norm(self.vision_projection(visual_features) + self.action_encoder(actions)
                            + self.state_encoder(state) + self.time_encoder(time_features)
                            + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x, allowed)
        x = self.final_norm(x).masked_fill(~valid[..., None], 0)
        result = {name: head(x).masked_fill(~valid[..., None], 0) for name, head in self.heads.items()}
        result["hidden"] = x
        result["valid"] = valid
        result['action_available'] = self.action_availability.expand(b, t, -1)
        return result
