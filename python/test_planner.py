"""Tests for D3/D4 planner and decision engine (synthetic states, D7)."""

import unittest

from game_state import (
    DoorState,
    GameState,
    PedestalItemState,
    PlayerState,
    RunInfo,
    SpecialRockState,
)
from planner.decision_engine import DecisionEngine
from planner.goals import GoalType
from planner.rules import PlannerContext, RulePlanner, STAGE_DEPTHS_2


def make_state(
    *,
    stage: int = 1,
    room_type: int = 1,
    enemy_count: int = 0,
    doors: list | None = None,
    pedestal_items: list | None = None,
    special_rocks: list | None = None,
    num_coins: int = 0,
    hp_red: float = 6.0,
) -> GameState:
    return GameState(
        run=RunInfo(stage=stage, room_type=room_type),
        enemy_count=enemy_count,
        player=PlayerState(hp_red=hp_red, num_coins=num_coins),
        doors=doors or [],
        pedestal_items=pedestal_items or [],
        special_rocks=special_rocks or [],
    )


class RulePlannerTests(unittest.TestCase):
    def test_combat_dominates_everything(self):
        planner = RulePlanner()
        state = make_state(enemy_count=3, pedestal_items=[PedestalItemState(item_id=105)])
        self.assertEqual(planner.plan(state).goal_type, GoalType.CLEAR_ROOM)

    def test_marked_skull_on_depths_2_triggers_bomb_goal_with_bomb(self):
        planner = RulePlanner(PlannerContext(preserve_bombs=True))
        state = make_state(
            stage=STAGE_DEPTHS_2,
            special_rocks=[SpecialRockState(kind="marked_skull", variant=2, position=(0.3, 0.4))],
        )
        state = GameState(
            run=state.run,
            enemy_count=0,
            player=PlayerState(hp_red=6.0, num_bombs=1),
            special_rocks=state.special_rocks,
            pedestal_items=[],
            doors=[],
        )
        subgoal = planner.plan(state)
        self.assertEqual(subgoal.goal_type, GoalType.BOMB_TARGET)
        self.assertEqual(subgoal.target_position, (0.3, 0.4))

    def test_pedestal_item_triggers_collect(self):
        planner = RulePlanner()
        state = make_state(pedestal_items=[PedestalItemState(item_id=105, position=(100, 50))])
        self.assertEqual(planner.plan(state).goal_type, GoalType.COLLECT_TARGET)

    def test_no_doors_no_items_waits(self):
        planner = RulePlanner()
        self.assertEqual(planner.plan(make_state()).goal_type, GoalType.WAIT)

    def test_goal_vector_round_trips(self):
        planner = RulePlanner()
        vector = planner.goal_vector(
            make_state(pedestal_items=[PedestalItemState(item_id=105, position=(100, 50))])
        )
        self.assertEqual(len(vector), 12)  # 8 goal types + 4 values
        from planner.goals import subgoal_from_vector
        subgoal = subgoal_from_vector(vector)
        self.assertEqual(subgoal.goal_type, GoalType.COLLECT_TARGET)


class DecisionEngineTests(unittest.TestCase):
    def test_hard_constraint_blocks_low_health_deal(self):
        rules = {
            "devil_deal": {
                "hard_constraints": [
                    {"feature": "current_health_ratio", "op": ">=", "value": 0.5}
                ]
            }
        }
        engine = DecisionEngine(rules)
        state = make_state(hp_red=0.0)  # total_hp 0 -> ratio 1... use custom
        state = GameState(
            run=state.run,
            enemy_count=0,
            player=PlayerState(hp_red=0.0, hp_soul=0.0),
            doors=[],
        )
        allowed, checks, _ = engine.evaluate("devil_deal", state)
        self.assertFalse(allowed)
        self.assertEqual(checks[0].feature, "current_health_ratio")

    def test_unknown_deal_type_is_allowed(self):
        engine = DecisionEngine({})
        allowed, checks, features = engine.evaluate("shop_purchase", make_state())
        self.assertTrue(allowed)
        self.assertEqual(checks, [])
        self.assertTrue(hasattr(features, "resource_pressure"))

    def test_features_are_numeric(self):
        engine = DecisionEngine({})
        _, _, features = engine.evaluate("devil_deal", make_state())
        vector = features.to_vector()
        self.assertEqual(len(vector), 8)
        self.assertTrue(all(isinstance(v, (int, float)) for v in vector))


if __name__ == "__main__":
    unittest.main()
