"""CS2 controls v2: angular view intent and per-branch history validity."""

import math
from dataclasses import dataclass

ACTION_SCHEMA_VERSION = 'cs2-controls-v2'
BUTTONS = ('jump', 'crouch', 'walk', 'reload', 'use', 'attack', 'attack2', 'drop')
BRANCHES = ('forward', 'strafe', *BUTTONS, 'weapon_slot', 'yaw', 'pitch')
YAW_BINS = (-180., -90., -45., -20., -10., -5., -2., -1., -.5, -.2, -.05, 0., .05, .2, .5, 1., 2., 5., 10., 20., 45., 90., 180.)
PITCH_BINS = (-89., -45., -20., -10., -5., -2., -1., -.5, -.2, -.05, 0., .05, .2, .5, 1., 2., 5., 10., 20., 45., 89.)
ACTION_DIM = 22 + len(BRANCHES)


def angle_target(value, bins):
    if not math.isfinite(value) or not bins[0] <= value <= bins[-1]:
        raise ValueError('Angular label is outside the action range')
    index = min(range(len(bins)), key=lambda i: abs(bins[i] - value))
    residual = value - bins[index]
    neighbor = min(index + 1, len(bins)-1) if residual >= 0 else max(index-1, 0)
    if neighbor == index:
        neighbor = index - 1 if index else 1
    scale = abs(bins[index] - bins[neighbor]) / 2
    return index, residual / scale


def decode_angle(index, residual, bins):
    neighbor = min(index+1, len(bins)-1) if residual >= 0 else max(index-1, 0)
    if neighbor == index:
        neighbor = index-1 if index else 1
    return max(bins[0], min(bins[-1], bins[index] + residual * abs(bins[index]-bins[neighbor]) / 2))


@dataclass(frozen=True)
class CS2Action:
    forward: int = 0
    strafe: int = 0
    buttons: tuple[bool, ...] = (False,) * len(BUTTONS)
    weapon_slot: int = 0
    delta_yaw: float = 0.
    delta_pitch: float = 0.
    available: tuple[bool, ...] = (True,) * len(BRANCHES)

    def __post_init__(self):
        if self.forward not in range(3) or self.strafe not in range(3) or self.weapon_slot not in range(6):
            raise ValueError('Invalid movement or weapon slot')
        if len(self.buttons) != len(BUTTONS) or any(type(b) is not bool for b in self.buttons):
            raise ValueError('buttons must contain eight booleans')
        if len(self.available) != len(BRANCHES) or any(type(b) is not bool for b in self.available):
            raise ValueError('available must contain thirteen booleans')
        if not math.isfinite(self.delta_yaw) or not math.isfinite(self.delta_pitch):
            raise ValueError('Angular deltas must be finite')
        if abs(self.delta_yaw) > 180 or abs(self.delta_pitch) > 178:
            raise ValueError('Angular deltas exceed the supported interval')


def encode_previous_action(action: CS2Action | None) -> tuple[float, ...]:
    if action is None:
        return (0.,) * ACTION_DIM
    v = action.available
    return (tuple(float(action.forward == i and v[0]) for i in range(3))
            + tuple(float(action.strafe == i and v[1]) for i in range(3))
            + tuple(float(x and v[2+i]) for i,x in enumerate(action.buttons))
            + tuple(float(action.weapon_slot == i and v[10]) for i in range(6))
            + (action.delta_yaw / 180 if v[11] else 0., action.delta_pitch / 180 if v[12] else 0.)
            + tuple(float(x) for x in v))


def decode_action(outputs, *, batch=0, step=-1) -> CS2Action:
    def category(name):
        return int(outputs[name][batch, step].argmax().item())
    residuals = outputs['angle_residual'][batch, step].tanh().tolist()
    available=tuple(outputs['action_available'][batch,step].tolist()) if 'action_available' in outputs else (True,)*len(BRANCHES)
    return CS2Action(forward=category('forward_logits') if available[0] else 0, strafe=category('strafe_logits') if available[1] else 0,
                     buttons=tuple(bool(x) and available[2+i] for i,x in enumerate((outputs['button_logits'][batch,step] >= 0).tolist())),
                     weapon_slot=category('weapon_logits') if available[10] else 0,
                     delta_yaw=decode_angle(category('yaw_logits'), residuals[0], YAW_BINS) if available[11] else 0.,
                     delta_pitch=decode_angle(category('pitch_logits'), residuals[1], PITCH_BINS) if available[12] else 0.,available=available)
