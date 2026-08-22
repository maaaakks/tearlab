"""Compact RSSM world model (DreamerV3-style) for structured Isaac states.

Stage-2 component of DREAMERV4_update.md: learns latent dynamics
p(z_{t+1} | h_t, z_t, a_t) OFFLINE from recorded transitions (human-demo npz
episodes now, live-recorded agent data later). Deliberately compact — RTX
4070 solo budget — and native-MultiDiscrete: actions enter as flat one-hot.

Structure:
  - ObservationEncoder: grid CNN + player MLP + masked-mean entity encoder
    -> embedding (no SB3 dependency; standalone from network.py).
  - RSSMCore: GRU deterministic path + single categorical stochastic state
    sampled straight-through; prior p(z|h) and posterior q(z|h, emb).
  - WorldModel: encoder + core + light heads (grid/player reconstruction,
    reward, continue) and the combined loss used by the offline trainer.

KL uses free nats so the posterior can carry information the prior cannot
predict without being punished into collapse (standard Dreamer trick).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def one_hot_actions(actions: torch.Tensor, action_nvec) -> torch.Tensor:
    """(B|B,T, K) int64 MultiDiscrete -> float one-hot of width sum(nvec)."""
    parts = []
    offset = 0
    for k, n in enumerate(action_nvec):
        parts.append(F.one_hot(actions[..., k], n).float())
        offset += n
    return torch.cat(parts, dim=-1)


def soft_sample_straight_through(logits: torch.Tensor) -> torch.Tensor:
    """ST-Gumbel categorical sample: hard one-hot forward, soft gradients."""
    uniform = torch.rand_like(logits).clamp_min(1e-6)
    gumbel = -torch.log(-torch.log(uniform))
    y = F.softmax(logits + gumbel, dim=-1)
    index = y.argmax(dim=-1)
    hard = F.one_hot(index, logits.shape[-1]).to(y.dtype)
    return hard + y - y.detach()


class _MLP(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), nn.ELU()]
            last = h
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ObservationEncoder(nn.Module):
    """Standalone structured-observation encoder (grid/player/entities)."""

    def __init__(
        self,
        grid_channels: int = 10,
        grid_hw=(7, 13),
        player_features: int = 22,
        max_entities: int = 32,
        entity_dim: int = 16,
        embed_dim: int = 192,
    ):
        super().__init__()
        c, hgt, wid = grid_channels, grid_hw[0], grid_hw[1]
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(c, 32, 3, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ELU(),
            nn.Flatten(),
        )
        self.grid_fc = nn.Linear(64 * hgt * wid, 128)
        self.player_mlp = _MLP(player_features, [64], 64)
        self.entity_mlp = _MLP(entity_dim, [64], 64)
        self.out = nn.Linear(128 + 64 + 64, embed_dim)

    def forward(self, grid, player, entities):
        """grid (…,C,H,W), player (…,F), entities (…,E,D) — any leading
        dims (batch and/or time); the CNN path is flattened internally."""
        lead = grid.shape[:-3]
        g = self.grid_cnn(grid.reshape(-1, *grid.shape[-3:]))
        g = self.grid_fc(g).reshape(*lead, -1)
        p = self.player_mlp(player)
        mask = (entities.abs().sum(dim=-1, keepdim=True) > 0).to(entities.dtype)
        pooled = (self.entity_mlp(entities) * mask).sum(dim=-2) / (
            mask.sum(dim=-2) + 1e-6
        )
        return self.out(torch.cat([g, p, pooled], dim=-1))


class RSSMCore(nn.Module):
    """Deterministic GRU path + categorical stochastic state."""

    def __init__(self, embed_dim: int, action_dim: int, deter: int = 256,
                 stoch: int = 32, hidden: int = 256):
        super().__init__()
        self.deter = deter
        self.stoch = stoch
        self.cell = nn.GRUCell(stoch + action_dim, deter)
        self.prior_net = _MLP(deter, [hidden], stoch)
        self.posterior_net = _MLP(deter + embed_dim, [hidden], stoch)

    def initial_state(self, batch_size: int, device):
        return (
            torch.zeros(batch_size, self.deter, device=device),
            torch.zeros(batch_size, self.stoch, device=device),
        )

    def observe(self, embeds: torch.Tensor, actions_oh: torch.Tensor,
                resets: torch.Tensor | None = None):
        """Filter the posterior over a sequence.

        embeds/actions_oh/resets: (B, T, ·). resets[t]=1 marks a step whose
        PREDECESSOR ended an episode (carried state is cleared before t).
        Returns (hiddens, prior_logits, post_logits, samples), each
        (B, T, ·).
        """
        B, T, _ = embeds.shape
        device = embeds.device
        h, z = self.initial_state(B, device)
        hiddens, priors, posts, samples = [], [], [], []
        for t in range(T):
            if resets is not None:
                r = resets[:, t:t + 1]
                h = h * (1.0 - r)
                z = z * (1.0 - r)
            inp = torch.cat([z, actions_oh[:, t]], dim=-1)
            h = self.cell(inp, h)
            prior_logits = self.prior_net(h)
            post_logits = self.posterior_net(torch.cat([h, embeds[:, t]], dim=-1))
            z = soft_sample_straight_through(post_logits)
            hiddens.append(h)
            priors.append(prior_logits)
            posts.append(post_logits)
            samples.append(z)
        return (
            torch.stack(hiddens, dim=1),
            torch.stack(priors, dim=1),
            torch.stack(posts, dim=1),
            torch.stack(samples, dim=1),
        )

    def imagine(self, actions_oh: torch.Tensor, h0: torch.Tensor,
                z0: torch.Tensor):
        """Roll out the PRIOR only (no observations) — imagination rollouts
        for stage 3. actions_oh: (B, T, A). Returns hiddens and prior
        samples (B, T, ·)."""
        B, T, _ = actions_oh.shape
        h, z = h0, z0
        hiddens, samples = [], []
        for t in range(T):
            inp = torch.cat([z, actions_oh[:, t]], dim=-1)
            h = self.cell(inp, h)
            logits = self.prior_net(h)
            z = soft_sample_straight_through(logits)
            hiddens.append(h)
            samples.append(z)
        return torch.stack(hiddens, dim=1), torch.stack(samples, dim=1)


class WorldModel(nn.Module):
    """Encoder + RSSM + reconstruction/reward/continue heads + losses."""

    def __init__(
        self,
        action_nvec=(9, 9, 2, 2, 2),
        grid_channels: int = 10,
        grid_hw=(7, 13),
        player_features: int = 22,
        max_entities: int = 32,
        entity_dim: int = 16,
        embed_dim: int = 192,
        deter: int = 256,
        stoch: int = 32,
        hidden: int = 256,
        free_nats: float = 3.0,
        kl_weight: float = 1.0,
        recon_weight: float = 1.0,
        reward_weight: float = 1.0,
        continue_weight: float = 1.0,
    ):
        super().__init__()
        self.action_nvec = list(int(n) for n in action_nvec)
        self.action_dim = sum(self.action_nvec)
        self.free_nats = free_nats
        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.reward_weight = reward_weight
        self.continue_weight = continue_weight
        self.grid_hw = tuple(grid_hw)

        self.encoder = ObservationEncoder(
            grid_channels=grid_channels,
            grid_hw=grid_hw,
            player_features=player_features,
            max_entities=max_entities,
            entity_dim=entity_dim,
            embed_dim=embed_dim,
        )
        self.core = RSSMCore(
            embed_dim=embed_dim,
            action_dim=self.action_dim,
            deter=deter,
            stoch=stoch,
            hidden=hidden,
        )
        state_dim = deter + stoch
        self.grid_head = _MLP(state_dim, [hidden],
                              grid_channels * grid_hw[0] * grid_hw[1])
        self.player_head = _MLP(state_dim, [hidden], player_features)
        self.reward_head = _MLP(state_dim, [hidden], 1)
        self.continue_head = _MLP(state_dim, [hidden], 1)

    def compute_losses(self, batch: dict) -> dict[str, torch.Tensor]:
        """batch keys (all leading (B, T)):
        grid (C,H,W) float, player (F,) float, entities (E,D) float,
        action (K,) int64, reward float, done bool."""
        grids = batch["grid"]
        players = batch["player"]
        entities = batch["entities"]
        B, T = grids.shape[0], grids.shape[1]
        device = grids.device

        embeds = self.encoder(grids, players, entities)
        actions_oh = one_hot_actions(batch["action"], self.action_nvec)
        resets = batch["done"].float()
        # A reset clears the carried state BEFORE the step following it, so
        # shift the terminal flags right by one inside the window.
        shifted = torch.zeros_like(resets)
        shifted[:, 1:] = resets[:, :-1]

        hiddens, prior_logits, post_logits, samples = self.core.observe(
            embeds, actions_oh, shifted
        )

        # KL(q || p), summed over the categorical, free-nats clamped.
        q = F.log_softmax(post_logits, dim=-1)
        p = F.log_softmax(prior_logits, dim=-1)
        kl = torch.sum(
            torch.exp(q) * (q - p), dim=-1
        )  # (B, T)
        kl_clamped = torch.clamp(kl, min=self.free_nats)

        state = torch.cat([hiddens, samples], dim=-1)
        flat_state = state.reshape(B * T, -1)
        grid_pred = self.grid_head(flat_state).reshape(grids.shape)
        player_pred = self.player_head(flat_state).reshape(players.shape)
        reward_pred = self.reward_head(flat_state).squeeze(-1).reshape(B, T)
        cont_pred = self.continue_head(flat_state).squeeze(-1).reshape(B, T)

        losses = {
            "kl": kl.mean().detach(),
            "kl_loss": kl_clamped.mean(),
            "grid_recon": F.mse_loss(grid_pred, grids),
            "player_recon": F.mse_loss(player_pred, players),
            "reward": F.mse_loss(reward_pred, batch["reward"]),
            "continue": F.binary_cross_entropy_with_logits(
                cont_pred, batch["done"].float()
            ),
        }
        losses["total"] = (
            self.kl_weight * losses["kl_loss"]
            + self.recon_weight
            * (losses["grid_recon"] + losses["player_recon"])
            + self.reward_weight * losses["reward"]
            + self.continue_weight * losses["continue"]
        )
        return losses
