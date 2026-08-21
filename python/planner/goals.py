"""Subgoal interface: the contract between planner and learned controller.

The controller receives the subgoal as a fixed-size goal vector appended to
its observation, so observation-shape changes stay additive (existing combat
checkpoints remain transferable via the shared encoder).
"""

from dataclasses import dataclass, field
from enum import IntEnum


class GoalType(IntEnum):
    """Discrete subgoal types. The one-hot encoding index must match
    GOAL_VECTOR_DIM layout in the controller's goal encoder."""

    IDLE = 0
    NAVIGATE_TO = 1        # move to target position (door, pickup, item)
    CLEAR_ROOM = 2         # kill all enemies in current room
    AVOID = 3              # keep distance from target position
    BOMB_TARGET = 4        # place bomb at target position
    COLLECT_TARGET = 5     # walk over specific pickup/pedestal
    INTERACT = 6           # use active item / card / pill at target
    WAIT = 7               # hold position (e.g. before deal decision)


@dataclass(frozen=True)
class Subgoal:
    goal_type: GoalType
    target_position: tuple[float, float] | None = None
    target_entity_type: int | None = None   # EntityType for identity checks
    target_entity_variant: int | None = None
    timeout_steps: int | None = None        # planner re-evaluates after this

    def to_vector(self) -> list[float]:
        """Fixed-size numeric vector consumed by the controller.

        Layout: [one-hot(GoalType), norm_pos_x, norm_pos_y,
                 entity_type_norm, entity_variant_norm]
        """
        n_goal_types = len(GoalType)
        vec = [0.0] * n_goal_types
        vec[int(self.goal_type)] = 1.0
        tx, ty = self.target_position or (0.0, 0.0)
        vec += [tx, ty]
        vec += [
            (self.target_entity_type or 0) / 1000.0,
            (self.target_entity_variant or 0) / 100.0,
        ]
        return vec


GOAL_VECTOR_DIM = len(GoalType) + 4


def subgoal_from_vector(vec: list[float]) -> Subgoal:
    """Inverse of to_vector; used by tests and offline replay."""
    n = len(GoalType)
    goal_type = GoalType(int(max(range(n), key=lambda i: vec[i])))
    pos = (vec[n], vec[n + 1])
    etype = round(vec[n + 2] * 1000)
    evar = round(vec[n + 3] * 100)
    return Subgoal(
        goal_type=goal_type,
        target_position=pos if any(pos) else None,
        target_entity_type=etype or None,
        target_entity_variant=evar or None,
    )
