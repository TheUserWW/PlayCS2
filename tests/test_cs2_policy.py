import unittest

import torch

from Transformer import CS2Policy, CS2PolicyConfig
from cs2.actions import ACTION_DIM, CS2Action, encode_previous_action
from cs2.gsi import GSIStore
from cs2.state import STATE_DIM
from cs2.timeline import PolicyTimeline


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(42)
        self.model = CS2Policy(CS2PolicyConfig(n_embed=48, n_head=3, n_layer=2,
                                              vision_dim=32, block_size=4, dropout=0,
                                              image_height=32, image_width=48)).eval()

    def inputs(self, batch=2, steps=4):
        return dict(
            visual_features=torch.randn(batch, steps, 32),
            previous_actions=torch.randn(batch, steps, ACTION_DIM),
            state_values=torch.rand(batch, steps, STATE_DIM),
            state_available=torch.ones(batch, steps, STATE_DIM, dtype=torch.bool),
            gsi_age=torch.zeros(batch, steps), gsi_fresh=torch.ones(batch, steps, dtype=torch.bool),
            timestamps=torch.arange(steps, dtype=torch.float64).repeat(batch, 1) * 0.05 + 1234567,
        )

    def test_forward_and_backward_through_all_modalities(self):
        self.model.train()
        data = self.inputs()
        del data["visual_features"]
        data["images"] = torch.randint(256, (2, 4, 3, 40, 60), dtype=torch.uint8)
        out = self.model(**data)
        self.assertEqual(out["forward_logits"].shape, (2, 4, 3))
        self.assertEqual(out["yaw_logits"].shape, (2, 4, 23))
        self.assertEqual(out["value"].shape, (2, 4, 1))
        loss = sum(v.square().mean() for k, v in out.items() if k.endswith("logits") or k == "value")
        loss.backward()
        for module in (self.model.frame_encoder, self.model.action_encoder, self.model.state_encoder,
                       self.model.blocks, self.model.time_encoder):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters()))

    def test_future_cannot_change_past_but_history_changes_future(self):
        data = self.inputs()
        original = self.model.forward_features(**data)["hidden"]
        changed = {k: v.clone() for k, v in data.items()}
        changed["visual_features"][:, 2:] += 10
        changed["previous_actions"][:, 2:] += 10
        changed["state_values"][:, 2:] = 0
        result = self.model.forward_features(**changed)["hidden"]
        torch.testing.assert_close(original[:, :2], result[:, :2])
        changed = {k: v.clone() for k, v in data.items()}
        changed["previous_actions"][:, 0] += 10
        result = self.model.forward_features(**changed)["hidden"]
        self.assertFalse(torch.allclose(original[:, -1], result[:, -1]))

    def test_padding_and_packed_episode_reset(self):
        data = self.inputs()
        starts = torch.tensor([[True, False, True, False]]).repeat(2, 1)
        original = self.model.forward_features(**data, episode_starts=starts)["hidden"]
        changed = {k: v.clone() for k, v in data.items()}
        changed["visual_features"][:, :2] += 10
        changed["previous_actions"][:, :3] += 10  # Action at reset is ignored too.
        changed["timestamps"][:, :2] -= 100
        result = self.model.forward_features(**changed, episode_starts=starts)["hidden"]
        torch.testing.assert_close(original[:, 2:], result[:, 2:])
        isolated = {k: v[:, 2:] for k, v in data.items()}
        separate = self.model.forward_features(**isolated, episode_starts=starts[:, 2:])["hidden"]
        torch.testing.assert_close(original[:, 2:], separate)
        valid = torch.tensor([[True, True, False, False], [True, True, True, True]])
        padded = self.model.forward_features(**data, valid=valid)
        self.assertTrue(torch.isfinite(padded["hidden"]).all())
        self.assertEqual(padded["hidden"][0, 2:].abs().sum(), 0)
        prefix = self.model.forward_features(**{k: v[:1, :2] for k, v in data.items()})
        torch.testing.assert_close(padded["hidden"][:1, :2], prefix["hidden"])

    def test_missing_or_stale_state_values_are_ignored(self):
        data = self.inputs()
        data["gsi_fresh"].zero_()
        original = self.model.forward_features(**data)["hidden"]
        data["state_values"] += 100
        data["state_available"].zero_()
        result = self.model.forward_features(**data)["hidden"]
        torch.testing.assert_close(original, result)

    def test_timeline_matches_batch_and_slides_then_resets(self):
        timeline = PolicyTimeline(self.model, GSIStore())
        frames = torch.randint(256, (6, 3, 40, 60), dtype=torch.uint8)
        actions = [None] + [CS2Action(forward=1, delta_yaw=3) for _ in range(5)]
        for i in range(6):
            proposal, online = timeline.observe(frames[i], captured_at=100 + i * 0.05, previous_action=actions[i])
        self.assertIsInstance(proposal, CS2Action)
        data = self.inputs(batch=1)
        del data["visual_features"]
        data.update(images=frames[-4:][None], previous_actions=torch.tensor([[encode_previous_action(a) for a in actions[-4:]]]),
                    state_values=torch.zeros(1, 4, STATE_DIM), state_available=torch.zeros(1, 4, STATE_DIM, dtype=torch.bool),
                    gsi_age=torch.full((1, 4), 2.0), gsi_fresh=torch.zeros(1, 4, dtype=torch.bool),
                    timestamps=torch.tensor([[100 + i * 0.05 for i in range(2, 6)]], dtype=torch.float64))
        offline = self.model(**data)
        torch.testing.assert_close(online["hidden"], offline["hidden"], atol=2e-6, rtol=1e-5)
        timeline.reset()
        _, online = timeline.observe(frames[0], captured_at=200, previous_action=None)
        self.assertEqual(online["hidden"].shape[1], 1)

    def test_contract_errors_are_explicit(self):
        data = self.inputs()
        data["timestamps"][:, 2] = 0
        with self.assertRaises(ValueError):
            self.model.forward_features(**data)
        timeline = PolicyTimeline(self.model, GSIStore())
        with self.assertRaises(ValueError):
            timeline.observe(torch.zeros(3, 32, 48, dtype=torch.uint8), captured_at=100, previous_action=CS2Action())

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA device is unavailable")
    def test_cuda_optimizer_step(self):
        model = self.model.to("cuda").train()
        data = {key: value.to("cuda") for key, value in self.inputs(batch=1).items()}
        del data["visual_features"]
        data["images"] = torch.randint(256, (1, 4, 3, 40, 60), dtype=torch.uint8, device="cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.heads["yaw_logits"].weight.detach().clone()
        output = model(**data)
        loss = output["yaw_logits"].square().mean() + output["value"].square().mean()
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        self.assertTrue(torch.isfinite(loss))
        self.assertFalse(torch.equal(before, model.heads["yaw_logits"].weight))


if __name__ == "__main__":
    unittest.main()
