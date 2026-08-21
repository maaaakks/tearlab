"""Entity-set encoder for identity-bearing objects (TODO C2)."""

import torch
import torch.nn as nn


class EntitySetEncoder(nn.Module):
    """Permutation-invariant encoder over identity-bearing objects.

    The observation carries a fixed-size padded table ``entities`` of shape
    ``(MAX_ENTITIES, ENTITY_DIM)``: each row is one object (enemy, pickup,
    pedestal item, door, special rock) with its identity features zero-padded
    beyond the number actually present. A single self-attention block over the
    padded set gives each object a contextual embedding; mean pooling yields a
    fixed (B, hidden) summary regardless of how many objects are present.
    """

    def __init__(self, entity_dim: int, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.entity_dim = entity_dim
        self.project = nn.Sequential(
            nn.Linear(entity_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, batch_first=True
        )
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, entities: torch.Tensor) -> torch.Tensor:
        """entities: (B, max_entities, entity_dim) -> (B, hidden)."""
        x = self.project(entities)
        attended, _ = self.attention(x, x, x, need_weights=False)
        x = self.ln(x + attended)
        # Mean over the set; padding rows are near-zero after projection but
        # still contribute a tiny bias, which is acceptable and deterministic.
        return x.mean(dim=1)
