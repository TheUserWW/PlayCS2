"""Versioned, masked local-player features; GSI is asynchronous, not a frame clock."""

import math
from dataclasses import dataclass
from typing import Any


SCHEMA_VERSION = "cs2-local-gsi-v1"
NUMERIC_FIELDS = (
    ("health", ("player", "state", "health"), 100.0),
    ("armor", ("player", "state", "armor"), 100.0),
    ("helmet", ("player", "state", "helmet"), 1.0),
    ("defusekit", ("player", "state", "defusekit"), 1.0),
    ("flashed", ("player", "state", "flashed"), 255.0),
    ("smoked", ("player", "state", "smoked"), 255.0),
    ("burning", ("player", "state", "burning"), 255.0),
    ("money", ("player", "state", "money"), 16000.0),
    ("round_kills", ("player", "state", "round_kills"), 5.0),
    ("round_killhs", ("player", "state", "round_killhs"), 5.0),
    ("round_totaldmg", ("player", "state", "round_totaldmg"), 500.0),
    ("equip_value", ("player", "state", "equip_value"), 10000.0),
    ("round_index", ("map", "round"), 30.0),
    ("ct_score", ("map", "team_ct", "score"), 30.0),
    ("t_score", ("map", "team_t", "score"), 30.0),
)
CATEGORICAL_FIELDS = (
    ("team", ("player", "team"), ("T", "CT")),
    ("activity", ("player", "activity"), ("playing", "menu", "textinput")),
    ("round_phase", ("round", "phase"), ("freezetime", "live", "over")),
    ("map_phase", ("map", "phase"), ("warmup", "live", "intermission", "gameover")),
    ("bomb", ("round", "bomb"), ("planted", "exploded", "defused")),
)
WEAPON_TYPES = ("Knife", "Pistol", "Submachine Gun", "Rifle", "Shotgun",
                "SniperRifle", "Machine Gun", "Grenade", "C4", "StackableItem")
WEAPON_NAMES = (
    "weapon_knife", "weapon_knife_t", "weapon_glock", "weapon_hkp2000", "weapon_usp_silencer",
    "weapon_p250", "weapon_deagle", "weapon_revolver", "weapon_elite", "weapon_fiveseven",
    "weapon_tec9", "weapon_cz75a", "weapon_mac10", "weapon_mp9", "weapon_mp7", "weapon_mp5sd",
    "weapon_ump45", "weapon_p90", "weapon_bizon", "weapon_galilar", "weapon_famas",
    "weapon_ak47", "weapon_m4a1", "weapon_m4a1_silencer", "weapon_aug", "weapon_sg556",
    "weapon_ssg08", "weapon_awp", "weapon_g3sg1", "weapon_scar20", "weapon_nova",
    "weapon_xm1014", "weapon_sawedoff", "weapon_mag7", "weapon_m249", "weapon_negev",
    "weapon_taser", "weapon_c4", "weapon_flashbang", "weapon_hegrenade", "weapon_smokegrenade",
    "weapon_molotov", "weapon_incgrenade", "weapon_decoy",
)
FEATURE_NAMES = (
    tuple(item[0] for item in NUMERIC_FIELDS)
    + tuple(f"{name}:{value}" for name, _, values in CATEGORICAL_FIELDS for value in (*values, "unknown"))
    + ("ammo_clip", "ammo_clip_max", "ammo_reserve")
    + tuple(f"weapon_type:{value}" for value in (*WEAPON_TYPES, "unknown"))
    + tuple(f"weapon_name:{value}" for value in (*WEAPON_NAMES, "unknown"))
)
STATE_DIM = len(FEATURE_NAMES)


def nested(data: dict, path: tuple[str, ...]) -> Any:
    value = data
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


@dataclass(frozen=True)
class StateFeatures:
    values: tuple[float, ...]
    available: tuple[bool, ...]
    age_seconds: float
    fresh: bool


def encode_gsi(snapshot, *, at: float, max_age: float = 2.0) -> StateFeatures:
    """Encode only data received by `at` (monotonic seconds).

    Missing/stale fields have zero values AND false masks. Never copy `previously`,
    enemy/observer-only data, or a spectated player's health into local-player inputs.
    A true mask with value zero means an actual observed zero, e.g. death/no ammo.
    """
    if not math.isfinite(at) or not math.isfinite(max_age) or max_age <= 0:
        raise ValueError("at must be finite and max_age must be positive")
    age = max_age if snapshot is None else at - snapshot.received_at
    fresh = snapshot is not None and 0 <= age <= max_age
    data = snapshot.data if fresh else {}
    data = dict(data)
    owner = nested(data, ("provider", "steamid"))
    player = nested(data, ("player", "steamid"))
    if not owner or owner != player:
        data.pop("player", None)
    return encode_observation(data, age_seconds=max(0.0, age), fresh=fresh)


def encode_observation(data, *, age_seconds=0.0, fresh=True):
    """Encode an already selected local player (offline demos or validated GSI)."""
    if not fresh:
        data = {}
    values, masks = [], []

    def number(value, scale):
        valid = isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
        values.append(min(float(value) / scale, 1.0) if valid else 0.0)
        masks.append(valid)

    def category(value, choices):
        present = isinstance(value, str) and bool(value)
        selected = value if value in choices else "unknown"
        for choice in (*choices, "unknown"):
            values.append(float(present and choice == selected))
            masks.append(present)

    for _, path, scale in NUMERIC_FIELDS:
        number(nested(data, path), scale)
    for _, path, choices in CATEGORICAL_FIELDS:
        category(nested(data, path), choices)
    weapons = nested(data, ("player", "weapons"))
    active = next((w for w in weapons.values() if isinstance(w, dict) and w.get("state") == "active"), {}) if isinstance(weapons, dict) else {}
    for key, scale in (("ammo_clip", 150.0), ("ammo_clip_max", 150.0), ("ammo_reserve", 500.0)):
        number(active.get(key), scale)
    category(active.get("type"), WEAPON_TYPES)
    category(active.get("name"), WEAPON_NAMES)
    return StateFeatures(tuple(values), tuple(masks), age_seconds, fresh)
