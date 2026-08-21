"""Rule-based planner: RunMemory + knowledge base -> Subgoal stream (TODO D3).

This is the deterministic layer that turns explicit game knowledge into a
*current goal vector* for the single generalist policy. The network keeps the
contextual decision among valid choices; the planner only resolves facts that
would otherwise have to be re-discovered by chance (TODO D4/D5).
"""

from dataclasses import dataclass

from game_state import GameState
from planner.goals import GoalType, Subgoal

# RoomType enum values (Repentance). Conservative fallback when the game
# reports a value not in this table.
ROOM_DEFAULT = 1
ROOM_TREASURE = 4
ROOM_BOSS = 5
ROOM_SHOP = 2
ROOM_DEVIL = 14
ROOM_ANGEL = 15
ROOM_SECRET = 7
ROOM_SUPERSECRET = 8

# Floor stage ids: 1=Basement I, 2=Basement II, 6=Depths II, ...
STAGE_DEPTHS_2 = 6
STAGE_HOME = 12


@dataclass
class PlannerContext:
    """Enough deterministic facts for the planner to pick a subgoal."""

    objective: str = "THE_BEAST"
    preserve_angel: bool = False
    preserve_bombs: bool = False


class RulePlanner:
    """Minimal deterministic planner over serialized run state.

    Priority order (hard facts first):
      1. Survival: if a boss room has no enemies it is safe to proceed; if
         enemies remain and HP is low, clear/avoid dominates everything.
      2. The Beast route milestones (Depths II, marked skull, Mom, ascent...).
      3. Economy: shop when coins exist and a shop door is reachable.
      4. Explore unvisited rooms otherwise.
    """

    def __init__(self, context: PlannerContext | None = None):
        self.context = context or PlannerContext()

    def plan(self, state: GameState) -> Subgoal:
        run = state.run

        # 1. In-combat / low HP dominates.
        if state.enemy_count > 0:
            return Subgoal(goal_type=GoalType.CLEAR_ROOM)

        # 2. Route milestone: on Depths II look for the marked skull.
        if (
            self.context.objective == "THE_BEAST"
            and run.stage >= STAGE_DEPTHS_2
            and run.stage < STAGE_HOME
        ):
            skull = next(
                (r for r in state.special_rocks if r.kind == "marked_skull"),
                None,
            )
            if skull is not None:
                if self.context.preserve_bombs and state.player.num_bombs <= 0:
                    # Cannot afford to bomb now: go find a bomb (shop/explore).
                    return self._pick_door_or_explore(state)
                return Subgoal(
                    goal_type=GoalType.BOMB_TARGET,
                    target_position=skull.position,
                    target_entity_type=1000,
                    target_entity_variant=skull.variant,
                )

        # 3. Collectible in sight: treasure room or pedestal item.
        if state.pedestal_items:
            item = state.pedestal_items[0]
            return Subgoal(
                goal_type=GoalType.COLLECT_TARGET,
                target_position=item.position,
                target_entity_type=5,
            )

        # 4. Shop economy when a shop door is reachable and coins exist.
        if run.room_type == ROOM_SHOP and state.player.num_coins > 0:
            pickup = state.pickups[0] if state.pickups else None
            if pickup is not None:
                return Subgoal(
                    goal_type=GoalType.COLLECT_TARGET,
                    target_position=pickup.position,
                )

        # 5. Otherwise explore the first unlocked door or simply wait.
        return self._pick_door_or_explore(state)

    def _pick_door_or_explore(self, state: GameState) -> Subgoal:
        if state.doors:
            # Prefer unlocked/open doors by slot order (deterministic).
            door = next(
                (d for d in state.doors if not d.locked),
                state.doors[0],
            )
            return Subgoal(
                goal_type=GoalType.NAVIGATE_TO,
                target_position=door.position,
                target_entity_type=door.slot,
            )
        return Subgoal(goal_type=GoalType.WAIT)

    def goal_vector(self, state: GameState) -> list[float]:
        return self.plan(state).to_vector()
