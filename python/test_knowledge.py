"""Tests for D1 items.json / D2 routes / D6 planner knowledge (synthetic)."""

import json
import unittest
from pathlib import Path

from knowledge_base import (
    KEY_PIECE_IDS,
    KnowledgeBase,
)
from planner.rules import PlannerContext, RulePlanner, ROOM_ANGEL
from game_state import (
    AngelStatueState,
    GameState,
    PedestalItemState,
    PlayerState,
    RunInfo,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


class ItemsKnowledgeTests(unittest.TestCase):
    def test_items_json_is_valid_and_loadable(self):
        path = _repo_root() / "knowledge" / "items.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(data["version"], 1)
        self.assertIsInstance(data["items"], dict)
        self.assertIn("tag_registry", data)

    def test_knowledge_base_resolves_quality_and_tags(self):
        base = KnowledgeBase()
        # 238/239 = Key Pieces (quest items for MEGA_SATAN), 386 = D100 (avoid).
        self.assertTrue(base.is_quest_item(238))
        self.assertTrue(base.is_quest_item(239))
        self.assertTrue(base.is_key_piece(239))
        self.assertTrue(base.is_bad_to_take(386))
        self.assertIn("reroll", base.tags(105))
        self.assertGreaterEqual(base.quality(182), 0)

    def test_tags_bitset_is_stable_and_additive(self):
        base = KnowledgeBase()
        self.assertEqual(base.tags_bitset(999999), 0)
        self.assertGreater(base.tags_bitset(169), 0)

    def test_build_synergy_score_overlaps_owned_tags(self):
        base = KnowledgeBase()
        owned = [105]  # reroll/utility/active
        direct = base.build_synergy_score(169, owned)  # damage: 0 overlap
        self.assertEqual(direct, 0.0)
        # A reroll item overlaps with a reroll build.
        overlap = base.build_synergy_score(105, [105, 552])
        self.assertGreater(overlap, 0.0)


class RouteKnowledgeTests(unittest.TestCase):
    def test_routes_are_loaded(self):
        base = KnowledgeBase()
        for objective in ("THE_BEAST", "MEGA_SATAN", "SURVIVE", "MOM"):
            self.assertIsNotNone(base.route(objective), objective)

    def test_mega_satan_requires_key_pieces_and_preserves_angel(self):
        base = KnowledgeBase()
        self.assertTrue(base.route_requires_key_pieces("MEGA_SATAN"))
        self.assertTrue(base.route_preserves_angel("MEGA_SATAN"))
        milestones = base.route_milestones("MEGA_SATAN")
        self.assertTrue(any(
            m["success"].get("condition") == "key_pieces_owned" or
            m["success"].get("params", {}).get("item_ids") == [238]
            for m in milestones
        ))


class ItemAvoidanceTests(unittest.TestCase):
    def test_planner_skips_bad_item_when_knowledge_attached(self):
        """D6: the planner must not COLLECT an item the KB marks avoid."""
        kb = KnowledgeBase()
        planner = RulePlanner(PlannerContext(objective="SURVIVE"), knowledge=kb)
        state = GameState(
            run=RunInfo(room_type=1),
            enemy_count=0,
            player=PlayerState(),
            pedestal_items=[
                PedestalItemState(item_id=386, position=(0.3, 0.3)),  # avoid
                PedestalItemState(item_id=169, position=(0.6, 0.6)),  # good
            ],
        )
        subgoal = planner.plan(state)
        from planner.goals import GoalType
        self.assertEqual(subgoal.goal_type, GoalType.COLLECT_TARGET)
        self.assertEqual(subgoal.target_position, (0.6, 0.6))

    def test_planner_without_knowledge_keeps_legacy_get_first_item(self):
        planner = RulePlanner(PlannerContext(objective="SURVIVE"))
        state = GameState(
            run=RunInfo(room_type=1),
            enemy_count=0,
            player=PlayerState(),
            pedestal_items=[PedestalItemState(item_id=386, position=(0.3, 0.3))],
        )
        subgoal = planner.plan(state)
        self.assertEqual(subgoal.target_position, (0.3, 0.3))


class MegaSatanPlannerTests(unittest.TestCase):
    def test_angel_statue_targeted_when_key_pieces_missing(self):
        planner = RulePlanner(PlannerContext(objective="MEGA_SATAN"))
        state = GameState(
            run=RunInfo(room_type=ROOM_ANGEL),
            enemy_count=0,
            player=PlayerState(key_pieces_owned=0, num_bombs=1),
            angel_statues=[AngelStatueState(position=(0.5, 0.5))],
        )
        subgoal = planner.plan(state)
        self.assertEqual(subgoal.goal_type.name, "BOMB_TARGET")

    def test_key_piece_collected_when_on_pedestal(self):
        planner = RulePlanner(PlannerContext(objective="MEGA_SATAN"))
        state = GameState(
            run=RunInfo(room_type=ROOM_ANGEL),
            enemy_count=0,
            player=PlayerState(key_pieces_owned=1),
            pedestal_items=[PedestalItemState(item_id=239, position=(0.4, 0.6))],
        )
        subgoal = planner.plan(state)
        from planner.goals import GoalType
        self.assertEqual(subgoal.goal_type, GoalType.COLLECT_TARGET)


if __name__ == "__main__":
    unittest.main()
