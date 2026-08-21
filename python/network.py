import torch
import torch.nn as nn
import numpy as np
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from entity_encoder import EntitySetEncoder


class IsaacFeatureExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for Isaac observations (TODO C3).

    Fuses four streams into a single policy embedding:
      - grid  : CNN over the local room grid
      - player: MLP over the normalized player vector
      - map   : CNN over the RunMemory room-graph summary (optional)
      - entities: permutation-invariant entity-set encoder over enemies,
                  pickups, prices, doors and interactive rocks (optional,
                  created only when the observation space carries an
                  ``entities`` key so older checkpoints keep their keys).
    """

    def __init__(self, observation_space: spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        grid_shape = observation_space["grid"].shape  # (C, H, W)
        player_shape = observation_space["player"].shape  # (N,)

        n_channels = grid_shape[0]

        # CNN for grid observations — no pooling, fewer channels
        # 32 channels × 7 × 13 = 2,912 after flatten (full spatial info preserved)
        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute CNN output size
        with torch.no_grad():
            sample = torch.zeros(1, *grid_shape)
            cnn_out_size = self.cnn(sample).shape[1]

        # MLP for player state
        self.player_mlp = nn.Sequential(
            nn.Linear(player_shape[0], 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        combined_size = cnn_out_size + 64

        # Optional floor-map branch (RunMemory summary). Created only when the
        # observation space carries a "map" key, so checkpoints trained without
        # map memory keep identical state_dict keys.
        self.has_map = "map" in observation_space.spaces
        if self.has_map:
            map_shape = observation_space["map"].shape  # (C, 9, 13)
            self.map_cnn = nn.Sequential(
                nn.Conv2d(map_shape[0], 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            with torch.no_grad():
                sample = torch.zeros(1, *map_shape)
                map_out_size = self.map_cnn(sample).shape[1]
            self.map_fc = nn.Sequential(nn.Linear(map_out_size, 32), nn.ReLU())
            combined_size += 32

        # Optional entity-set branch (TODO C3): merge identity-bearing objects
        # into the same embedding.
        self.has_entities = "entities" in observation_space.spaces
        if self.has_entities:
            entity_shape = observation_space["entities"].shape  # (MAX, D)
            self.entity_encoder = EntitySetEncoder(entity_shape[1])
            combined_size += 64

        # Final projection to features_dim
        self.fc = nn.Sequential(
            nn.Linear(combined_size, features_dim),
            nn.ReLU(),
        )

        self._features_dim = features_dim

    def forward(self, observations: dict) -> torch.Tensor:
        grid = observations["grid"]
        player = observations["player"]

        cnn_out = self.cnn(grid)
        player_out = self.player_mlp(player)

        parts = [cnn_out, player_out]
        if self.has_map and "map" in observations:
            map_out = self.map_fc(self.map_cnn(observations["map"]))
            parts.append(map_out)
        if self.has_entities and "entities" in observations:
            entity_out = self.entity_encoder(observations["entities"])
            parts.append(entity_out)

        combined = torch.cat(parts, dim=1)
        return self.fc(combined)
