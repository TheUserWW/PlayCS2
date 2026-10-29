"""Inference adapter joining screenshot times, executed actions and GSI snapshots."""

from collections import deque
import math

import torch

from .actions import CS2Action, decode_action, encode_previous_action
from .state import encode_gsi


class PolicyTimeline:
    """Bounded timeline; CNN features are cached, Transformer context recomputed.

    Call reset() on round/map changes, respawn, model/weight changes, or capture
    discontinuities. Call observe only after model.eval(). No KV cache is used:
    learned window-relative positions are recomputed consistently at each step.
    """

    def __init__(self, policy, gsi_store, *, max_gsi_age=2.0):
        if not math.isfinite(max_gsi_age) or max_gsi_age <= 0:
            raise ValueError("max_gsi_age must be positive")
        self.policy = policy
        self.store = gsi_store
        self.max_gsi_age = max_gsi_age
        self._history = deque(maxlen=policy.config.block_size)

    def reset(self):
        self._history.clear()

    @torch.no_grad()
    def observe(self, frame, *, captured_at: float, previous_action: CS2Action | None):
        """frame: uint8 RGB [3,H,W], captured_at: time.monotonic() screenshot time.

        previous_action must reflect input actually executed since the previous
        screenshot, not merely the last proposal. First observation must use None.
        """
        if self.policy.training:
            raise RuntimeError("Call policy.eval() before creating inference history")
        if not self._history and previous_action is not None:
            raise ValueError("First observation after reset must have previous_action=None")
        if self._history and (previous_action is None or captured_at <= self._history[-1][1]):
            raise ValueError("Provide an executed previous action and increasing capture timestamps")
        if frame.ndim != 3 or frame.shape[0] != 3:
            raise ValueError("frame must be RGB [3,H,W]")
        state = encode_gsi(self.store.at_or_before(captured_at), at=captured_at, max_age=self.max_gsi_age)
        device = next(self.policy.parameters()).device
        features = self.policy.encode_frames(frame.to(device)[None, None])[0, 0]
        row = (features, captured_at, encode_previous_action(previous_action), state, not self._history)
        self._history.append(row)
        rows = list(self._history)

        def tensor(values, dtype=torch.float32):
            return torch.tensor(values, dtype=dtype, device=device)[None]

        result = self.policy.forward_features(
            torch.stack([r[0] for r in rows])[None],
            tensor([r[2] for r in rows]), tensor([r[3].values for r in rows]),
            tensor([r[3].available for r in rows], torch.bool),
            tensor([r[3].age_seconds for r in rows]), tensor([r[3].fresh for r in rows], torch.bool),
            tensor([r[1] for r in rows], torch.float64),
            episode_starts=tensor([r[4] for r in rows], torch.bool),
        )
        return decode_action(result), result
