"""Knowledge base loader (TODO D1/D6).

Loads the machine-readable knowledge from ``knowledge/`` and turns it into
cheap numeric facts for the planner / decision engine. Nothing here is learned;
it is deterministic, inspectable and versioned, per the project's
"connaissance observable" principle.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import yaml

from utils import get_repo_root

# Item ids that must be collected/used for specific routes.
KEY_PIECE_IDS = (238, 239)
POLAROID_IDS = (327, 328)

# Quest items that are never "good" to skip for their route (D6).
QUEST_ITEM_IDS = {
    238: "mega_satan_key_piece_1",
    239: "mega_satan_key_piece_2",
    327: "polaroid",
    328: "negative",
    549: "dads_note",
}

TAG_POSITIVE_PRIORITY = (
    "damage",
    "defensive",
    "utility",
    "quest",
    "key_piece",
    "zone_access",
    "the_beast",
    "mega_satan",
    "hearts",
    "rewind",
    "vision",
    "movement",
    "active",
)


class ItemRecord:
    __slots__ = ("name", "quality", "tags", "transformations", "avoid")

    def __init__(self, data: dict):
        self.name = str(data.get("name", ""))
        self.quality = int(data.get("quality", 0))
        self.tags = frozenset(data.get("tags", []))
        self.transformations = tuple(data.get("transformations", []))
        self.avoid = bool(data.get("avoid", False))


class KnowledgeBase:
    """Deterministic facts about items, routes and decision rules."""

    def __init__(self, repo_root: Path | None = None):
        root = repo_root or get_repo_root()
        self.root = Path(root)
        self.items: dict[int, ItemRecord] = {}
        self.transformations: dict[str, dict] = {}
        self.routes: dict[str, dict] = {}
        self.decision_rules: dict[str, dict] = {}
        self._load_items()
        self._load_routes()
        self._load_decision_rules()

    # -- loading --------------------------------------------------------

    def _load_items(self) -> None:
        path = self._knowledge_path("items.json")
        if path is None:
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        for item_id, record in (data.get("items") or {}).items():
            self.items[int(item_id)] = ItemRecord(record)
        self.transformations = data.get("transformations") or {}

    def _load_routes(self) -> None:
        routes_dir = self.root / "knowledge" / "routes"
        if not routes_dir.is_dir():
            return
        for path in sorted(routes_dir.glob("*.yaml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            objective = data.get("objective")
            if objective:
                self.routes[str(objective)] = data

    def _load_decision_rules(self) -> None:
        rules_dir = self.root / "knowledge" / "decision_rules"
        if not rules_dir.is_dir():
            return
        for path in sorted(rules_dir.glob("*.yaml")):
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            self.decision_rules.update(data)

    def _knowledge_path(self, filename: str) -> Path | None:
        path = self.root / "knowledge" / filename
        return path if path.is_file() else None

    # -- item facts -----------------------------------------------------

    @lru_cache(maxsize=4096)
    def get(self, item_id: int) -> ItemRecord | None:
        return self.items.get(int(item_id))

    def quality(self, item_id: int) -> float:
        record = self.get(item_id)
        return float(record.quality) if record else 0.0

    @lru_cache(maxsize=4096)
    def tags(self, item_id: int) -> frozenset[str]:
        record = self.get(item_id)
        return record.tags if record else frozenset()

    def is_quest_item(self, item_id: int) -> bool:
        return int(item_id) in QUEST_ITEM_IDS

    def quest_role(self, item_id: int) -> str | None:
        return QUEST_ITEM_IDS.get(int(item_id))

    def is_key_piece(self, item_id: int) -> bool:
        return int(item_id) in KEY_PIECE_IDS

    def is_zone_access(self, item_id: int) -> bool:
        return "zone_access" in self.tags(item_id)

    def is_bad_to_take(self, item_id: int) -> bool:
        record = self.get(item_id)
        return bool(record.avoid) if record else False

    def has_tag(self, item_id: int, tag: str) -> bool:
        return tag in self.tags(item_id)

    def tags_bitset(self, item_id: int) -> int:
        """Map an item's tags onto a stable bit register.

        Bit positions are derived from TAG_POSITIVE_PRIORITY plus any tag not
        listed there (appended in sorted order), so the network receives a
        deterministic, additive feature. Returns 0 for unknown items.
        """
        bits = 0
        for tag in self.tags(item_id):
            if tag in TAG_POSITIVE_PRIORITY:
                bits |= 1 << TAG_POSITIVE_PRIORITY.index(tag)
        return bits

    def transformation_tags(self, owned_item_ids: list[int]) -> frozenset[str]:
        owned = set(int(i) for i in owned_item_ids)
        result: set[str] = set()
        for spec in self.transformations.values():
            if set(spec.get("item_ids", [])) & owned:
                result.update(spec.get("tags", []))
        return frozenset(result)

    # -- synergies ------------------------------------------------------

    def build_synergy_score(
        self, candidate_id: int, owned_item_ids: list[int]
    ) -> float:
        """0..1 overlap between the candidate's tags and the owned build.

        Costly-tag items and transformations count as synergy too, so the
        decision feature reflects "how well does this item fit the current
        build" without hard-coding every interaction.
        """
        if len(owned_item_ids) == 0:
            return 0.0
        owned_tags: set[str] = set()
        for owned in owned_item_ids:
            owned_tags.update(self.tags(int(owned)))
        owned_tags.update(self.transformation_tags(owned_item_ids))
        candidate_tags = self.tags(int(candidate_id))
        if not candidate_tags:
            return 0.0
        return len(candidate_tags & owned_tags) / len(candidate_tags)

    # -- route facts ----------------------------------------------------

    def route(self, objective: str) -> dict | None:
        return self.routes.get(objective)

    def route_milestones(self, objective: str) -> list[dict]:
        route = self.route(objective)
        return list(route.get("milestones", [])) if route else []

    def route_requires_key_pieces(self, objective: str) -> bool:
        """MEGA_SATAN-style routes require both Key Pieces."""
        route = self.route(objective)
        if not route:
            return False
        milestones = route.get("milestones", [])
        for milestone in milestones:
            if milestone.get("success", {}).get("condition") == "key_pieces_owned":
                return True
        return bool(route.get("requires_key_pieces"))

    def route_max_deals(self, objective: str) -> int | None:
        """How many Devil Deals a route allows (-1 = none, None = unrestricted)."""
        deals = self.decision_rules.get("devil_deal", {})
        overrides = deals.get("route_overrides", {}) or {}
        override = overrides.get(objective, {})
        if "max_deals" in override:
            value = override["max_deals"]
            return int(value) if value is not None else None
        return None

    def route_preserves_angel(self, objective: str) -> bool:
        max_deals = self.route_max_deals(objective)
        return max_deals is not None and max_deals <= 0
