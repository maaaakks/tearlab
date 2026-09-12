"""Small action-conditioned ensemble for a bounded synthetic control experiment.

The observation contract is D1's two instrumented frames. Object pools are
computed independently per frame: a recycled projectile slot is not treated as
the same tracked object. Coordinates and action IDs provide an interface, not
tactical rules. Every parameter starts from a fresh random initialization.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class _DuelingHead(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.value = nn.Linear(width, 1)
        self.advantage = nn.Linear(width, 45)

    def forward(self, state):
        advantage = self.advantage(state)
        return self.value(state) + advantage - advantage.mean(-1, keepdim=True)


class RapidQ(nn.Module):
    """Shared object representation, separate Q heads and optional dynamics.

    ``previous_action`` is the last requested command, or 45 at episode start.
    Q values have shape [batch, ensemble heads, 45]. Auxiliary predictions are
    [dx, dy, total enemy HP change / 48, contact logit] for each future command;
    the contact output is a logit for BCEWithLogits, not a probability. The
    auxiliary receives only the initial history and commands actually executed.
    It never consumes future observations or selects the control action.
    """

    def __init__(self, heads: int = 3, auxiliary: bool = True):
        super().__init__()
        if heads < 1:
            raise ValueError('At least one Q head is required')
        self.heads = heads
        self.auxiliary = auxiliary
        self.enemy_encoder = nn.Sequential(
            nn.Linear(6, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())
        self.projectile_encoder = nn.Sequential(
            nn.Linear(5, 32), nn.SiLU(), nn.Linear(32, 32), nn.SiLU())
        # Each frame: player/room coordinates + three sum/max object pools.
        self.trunk = nn.Sequential(
            nn.Linear(2 * (4 + 3 * 64) + 46, 128), nn.SiLU(),
            nn.Linear(128, 128), nn.SiLU())
        self.q_heads = nn.ModuleList(_DuelingHead(128) for _ in range(heads))
        if auxiliary:
            self.dynamics = nn.GRUCell(45, 128)
            self.consequences = nn.Linear(128, 4)

    @staticmethod
    def _pool(encoder, raw, player):
        present = raw[..., -1].bool()
        # Remove absent slot contents before encoding, including stale values.
        raw = torch.where(present[..., None], raw, 0.)
        relative = raw[..., :2] - player[..., None, :]
        inputs = torch.cat((raw[..., :2], relative, raw[..., 2:]), -1)
        encoded = encoder(inputs)
        total = torch.where(present[..., None], encoded, 0.).sum(-2)
        maximum = torch.where(present[..., None], encoded, -torch.inf).amax(-2)
        maximum = torch.where(present.any(-1, keepdim=True), maximum, 0.)
        return torch.cat((total, maximum), -1)

    def _state(self, history, previous_action):
        batch = len(history)
        player = history[..., :2]
        enemies = history[..., 4:20].reshape(batch, 2, 4, 4)
        hostile = history[..., 20:116].reshape(batch, 2, 32, 3)
        friendly = history[..., 116:140].reshape(batch, 2, 8, 3)
        frames = torch.cat((
            history[..., :4],
            self._pool(self.enemy_encoder, enemies, player),
            self._pool(self.projectile_encoder, hostile, player),
            self._pool(self.projectile_encoder, friendly, player)), -1)
        previous = F.one_hot(previous_action, 46).to(history.dtype)
        return self.trunk(torch.cat((frames.flatten(1), previous), -1))

    def forward(self, history, previous_action):
        state = self._state(history, previous_action)
        return torch.stack([head(state) for head in self.q_heads], 1)

    def auxiliary_predictions(self, history, previous_action, actions):
        if not self.auxiliary:
            raise ValueError('Auxiliary dynamics are disabled for this ablation')
        state = self._state(history, previous_action)
        predictions = []
        for action in actions.unbind(1):
            state = self.dynamics(F.one_hot(action, 45).to(state.dtype), state)
            predictions.append(self.consequences(state))
        if not predictions:
            return state.new_empty(len(state), 0, 4)
        return torch.stack(predictions, 1)
