import unittest

import numpy as np
import torch

from rssm import (
    ObservationEncoder,
    RSSMCore,
    WorldModel,
    one_hot_actions,
    soft_sample_straight_through,
)

ACTION_NVEC = [9, 9, 2, 2, 2]


def _synthetic_batch(batch: int = 4, seq_len: int = 8, seed: int = 0):
    rng = np.random.default_rng(seed)
    return {
        "grid": rng.random((batch, seq_len, 10, 7, 13), dtype=np.float32),
        "player": (rng.random((batch, seq_len, 22), dtype=np.float32) - 0.5),
        "entities": (rng.random((batch, seq_len, 32, 16), dtype=np.float32)
                     * 2 - 1),
        "action": np.stack([
            np.stack([rng.integers(0, n) for n in ACTION_NVEC])
            for _ in range(batch * seq_len)
        ]).reshape(batch, seq_len, len(ACTION_NVEC)).astype(np.int64),
        "reward": rng.normal(0, 1, (batch, seq_len)).astype(np.float32),
        "done": np.zeros((batch, seq_len), dtype=bool),
    }


class ShapeTests(unittest.TestCase):
    def test_one_hot_actions(self):
        actions = torch.tensor([[1, 2, 0, 1, 0], [8, 0, 1, 0, 1]])
        one_hot = one_hot_actions(actions, ACTION_NVEC)
        self.assertEqual(one_hot.shape, (2, sum(ACTION_NVEC)))
        self.assertAlmostEqual(one_hot[0, 1].item(), 1.0)
        # Row 1 pillcard=1 -> last block start is sum(nvec[:4]); value 1
        # lands on its second class.
        self.assertAlmostEqual(one_hot[1, sum(ACTION_NVEC[:4]) + 1].item(), 1.0)

    def test_straight_through_sample_is_valid_distribution(self):
        torch.manual_seed(0)
        logits = torch.randn(5, 16, requires_grad=True)
        sample = soft_sample_straight_through(logits)
        sums = sample.sum(dim=-1)
        self.assertTrue(torch.allclose(sums, torch.ones(5)))
        # Straight-through: gradient must flow through the sampled value.
        sample.sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_observe_sequence_shapes(self):
        torch.manual_seed(0)
        core = RSSMCore(embed_dim=16, action_dim=sum(ACTION_NVEC),
                        deter=32, stoch=8, hidden=32)
        embeds = torch.randn(3, 6, 16)
        actions = one_hot_actions(
            torch.randint(0, 2, (3, 6, len(ACTION_NVEC))), ACTION_NVEC
        )
        hiddens, priors, posts, samples = core.observe(embeds, actions)
        for tensor in (hiddens, priors, posts, samples):
            self.assertEqual(tensor.shape[:2], (3, 6))
        self.assertEqual(hiddens.shape[-1], 32)
        self.assertEqual(samples.shape[-1], 8)

    def test_imagine_rollout_shapes(self):
        torch.manual_seed(0)
        core = RSSMCore(embed_dim=16, action_dim=sum(ACTION_NVEC),
                        deter=32, stoch=8, hidden=32)
        h0 = torch.zeros(2, 32)
        z0 = torch.zeros(2, 8)
        actions = one_hot_actions(
            torch.randint(0, 2, (2, 5, len(ACTION_NVEC))), ACTION_NVEC
        )
        hiddens, samples = core.imagine(actions, h0, z0)
        self.assertEqual(hiddens.shape, (2, 5, 32))
        self.assertEqual(samples.shape, (2, 5, 8))

    def test_world_model_losses_are_finite(self):
        torch.manual_seed(0)
        model = WorldModel(action_nvec=ACTION_NVEC, deter=32, stoch=8,
                           hidden=32, embed_dim=24)
        losses = model.compute_losses(
            {k: torch.as_tensor(v) for k, v in _synthetic_batch().items()}
        )
        expected = {"total", "kl", "kl_loss", "grid_recon", "player_recon",
                    "reward", "continue"}
        self.assertEqual(set(losses), expected)
        for name, value in losses.items():
            self.assertTrue(torch.isfinite(value), f"{name} not finite")
        losses["total"].backward()


class ToyDynamicsLearningTest(unittest.TestCase):
    def test_loss_decreases_on_deterministic_dynamics(self):
        """On a toy env where the player position follows the move action
        exactly, the world model must learn to predict it — the offline
        equivalent of the S1 probe."""
        torch.manual_seed(0)
        np.random.seed(0)

        deltas = {
            0: (0.0, 0.0), 1: (0.05, 0.0), 2: (-0.05, 0.0),
            3: (0.0, 0.05), 4: (0.0, -0.05),
        }

        def make_episode(length=64):
            rng = np.random.default_rng()
            pos = np.array([rng.uniform(0.2, 0.8),
                            rng.uniform(0.2, 0.8)], dtype=np.float32)
            grids, players, actions, rewards, dones = [], [], [], [], []
            for t in range(length):
                a = int(rng.integers(0, 5))
                grids.append(np.random.random((10, 7, 13))
                             .astype(np.float32))
                players.append(np.concatenate([
                    pos.copy(), np.zeros(20, dtype=np.float32)]))
                actions.append((a, 0, 0, 0, 0))
                rewards.append(np.float32(-pos.sum()))
                dones.append(False)
                if a in deltas:
                    pos = np.clip(pos + np.asarray(deltas[a]), 0.0, 1.0)
                    pos = pos.astype(np.float32)
            return {
                "grid": np.stack(grids),
                "player": np.stack(players),
                "entities": np.zeros((length, 32, 16), dtype=np.float32),
                "action": np.asarray(actions, dtype=np.int64),
                "reward": np.asarray(rewards, dtype=np.float32),
                "done": np.asarray(dones, dtype=bool),
            }

        episodes = [make_episode() for _ in range(8)]

        class TinyStore:
            seq_len = 16

            def __init__(self, eps):
                self.episodes = eps

            def sample_windows(self, indices):
                import numpy as np_local
                out = {k: [] for k in episodes[0]}
                for i in indices:
                    ep = self.episodes[i % len(self.episodes)]
                    s = np_local.random.randint(
                        0, len(ep["reward"]) - self.seq_len + 1)
                    sl = slice(int(s), int(s) + self.seq_len)
                    for k in out:
                        out[k].append(ep[k][sl])
                return {
                    k: np.stack(v).astype(
                        np.bool_ if k == "done" else
                        np.int64 if k == "action" else np.float32
                    )
                    for k, v in out.items()
                }

        store = TinyStore(episodes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = WorldModel(action_nvec=ACTION_NVEC, deter=64, stoch=8,
                           hidden=64, embed_dim=32).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

        def avg_player_recon(steps):
            model.eval()
            with torch.no_grad():
                total = 0.0
                for i in range(steps):
                    batch = {
                        k: torch.as_tensor(v, device=device)
                        for k, v in store.sample_windows(range(4)).items()
                    }
                    total += float(model.compute_losses(batch)["player_recon"])
            model.train()
            return total / steps

        first = avg_player_recon(3)
        for _ in range(120):
            batch = {
                k: torch.as_tensor(v, device=device)
                for k, v in store.sample_windows(range(8)).items()
            }
            losses = model.compute_losses(batch)
            optimizer.zero_grad()
            losses["total"].backward()
            optimizer.step()
        last = avg_player_recon(3)

        self.assertLess(last, first,
                        f"player recon did not improve: {first} -> {last}")
        self.assertLess(last / max(first, 1e-8), 0.7)


if __name__ == "__main__":
    unittest.main()
