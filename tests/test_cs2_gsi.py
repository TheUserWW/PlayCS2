import copy
import http.client
import json
import unittest

from cs2.gsi import GSIReceiver, GSISnapshot, GSIStore
from cs2.state import FEATURE_NAMES, STATE_DIM, encode_gsi


def payload(health=80):
    return {
        "provider": {"appid": 730, "steamid": "local", "timestamp": 100},
        "player": {"steamid": "local", "team": "CT", "activity": "playing",
                   "state": {"health": health, "armor": 0, "helmet": False},
                   "weapons": {"weapon_0": {"state": "active", "name": "weapon_ak47",
                                            "type": "Rifle", "ammo_clip": 0, "ammo_clip_max": 30}}},
        "map": {"name": "de_dust2", "round": 3}, "round": {"phase": "live"},
    }


class StateTests(unittest.TestCase):
    def test_zero_missing_unknown_and_local_identity(self):
        state = encode_gsi(GSISnapshot(payload(), 10.0, 1), at=10.1)
        self.assertEqual(len(state.values), STATE_DIM)
        for key in ("armor", "helmet", "ammo_clip"):
            i = FEATURE_NAMES.index(key)
            self.assertEqual(state.values[i], 0)
            self.assertTrue(state.available[i])
        self.assertFalse(state.available[FEATURE_NAMES.index("money")])
        self.assertEqual(state.values[FEATURE_NAMES.index("weapon_name:weapon_ak47")], 1)
        other = payload()
        other["player"]["steamid"] = "spectated"
        state = encode_gsi(GSISnapshot(other, 10.0, 1), at=10.1)
        self.assertFalse(state.available[FEATURE_NAMES.index("health")])
        self.assertTrue(state.available[FEATURE_NAMES.index("round_phase:live")])
        other = payload()
        other["player"]["weapons"]["weapon_0"]["name"] = "weapon_future"
        state = encode_gsi(GSISnapshot(other, 10.0, 1), at=10.1)
        self.assertEqual(state.values[FEATURE_NAMES.index("weapon_name:unknown")], 1)

    def test_stale_missing_and_future_have_no_valid_fields(self):
        for snapshot, now in ((None, 10), (GSISnapshot(payload(), 10, 1), 13), (GSISnapshot(payload(), 10, 1), 9)):
            state = encode_gsi(snapshot, at=now)
            self.assertFalse(state.fresh)
            self.assertFalse(any(state.available))
            self.assertFalse(any(state.values))

    def test_store_uses_current_snapshot_and_never_future(self):
        store = GSIStore(history_size=2)
        first = payload()
        store.publish(first, received_at=10)
        first["player"]["state"]["health"] = 1
        second = payload(50)
        second["player"]["state"].pop("armor")
        second["previously"] = {"player": {"state": {"armor": 100}}}
        second["auth"] = {"token": "secret"}
        store.publish(second, received_at=11)
        self.assertIsNone(store.at_or_before(9))
        self.assertEqual(store.at_or_before(10.5).data["player"]["state"]["health"], 80)
        latest = store.latest()
        self.assertNotIn("armor", latest.data["player"]["state"])
        self.assertNotIn("previously", latest.data)
        self.assertNotIn("auth", latest.data)
        latest.data.clear()
        self.assertTrue(store.latest().data)
        self.assertIsNone(store.wait_for_update(2, timeout=0.001))

    def test_reject_wrong_game_and_non_objects(self):
        for invalid in ([], {}, {"provider": {"appid": 440}}):
            with self.assertRaises(ValueError):
                GSIStore().publish(invalid)


class ReceiverTests(unittest.TestCase):
    def test_real_http_post_auth_invalid_payload_and_shutdown(self):
        receiver = GSIReceiver(port=0, token="test-token")
        with receiver:
            host, port = receiver.address

            def post(body, path="/"):
                connection = http.client.HTTPConnection(host, port, timeout=3)
                try:
                    connection.request("POST", path, body, {"Content-Type": "application/json"})
                    response = connection.getresponse()
                    response.read()
                    return response.status
                finally:
                    connection.close()

            data = payload()
            self.assertEqual(post(json.dumps(data)), 403)
            self.assertEqual(post("{"), 400)
            data["auth"] = {"token": "test-token"}
            self.assertEqual(post(json.dumps(data), "/wrong"), 404)
            self.assertEqual(post(json.dumps(data)), 200)
            self.assertEqual(receiver.store.wait_for_update(timeout=0.1).sequence, 1)
            invalid = copy.deepcopy(data)
            invalid["provider"]["appid"] = 440
            self.assertEqual(post(json.dumps(invalid)), 400)
        receiver.close()  # Idempotent.
        with receiver:  # Restart after closing is supported.
            self.assertIsNotNone(receiver.address)


if __name__ == "__main__":
    unittest.main()
