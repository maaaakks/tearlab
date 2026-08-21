"""Decision engine skeleton (TODO D4/D5).

Hard constraints (deterministic facts) and utility features (context) both
come from the knowledge base; the network makes the actual choice when more
than one option is valid. Nothing here creates a separate policy — the
planner emits features that the generalist policy consumes.
"""

from dataclasses import dataclass

from game_state import GameState


@dataclass(frozen=True)
class ConstraintCheck:
    feature: str
    passed: bool
    value: float
    op: str
    threshold: float


@dataclass
class DecisionFeatures:
    """Numeric features exposed to the network for a decision point."""

    heart_price: float = 0.0
    current_health_ratio: float = 1.0
    item_quality: float = 0.0
    item_tags_bitset: int = 0
    build_synergy_score: float = 0.0
    deal_history_penalty: float = 0.0
    floor_stage: float = 0.0
    resource_pressure: float = 0.0

    def to_vector(self) -> list[float]:
        return [
            self.heart_price,
            self.current_health_ratio,
            self.item_quality,
            float(self.item_tags_bitset),
            self.build_synergy_score,
            self.deal_history_penalty,
            self.floor_stage,
            self.resource_pressure,
        ]


class DecisionEngine:
    """Applies hard constraints; leaves context among valid options learned."""

    def __init__(self, rules: dict | None = None):
        # rules: {deal_type: {"hard_constraints": [...], ...}}
        self.rules = rules or {}

    def evaluate(self, deal_type: str, state: GameState) -> tuple[bool, list[ConstraintCheck], DecisionFeatures]:
        """Return (allowed, checks, features) for a deal/decision point."""
        deal = self.rules.get(deal_type, {})
        checks = []
        features = DecisionFeatures()

        total_hp = state.player.total_hp
        if total_hp <= 0.0:
            # No hearts left: maximally unsafe, ratio 0.
            health_ratio = 0.0
        else:
            health_ratio = total_hp / max(
                1.0, state.player.hp_red + state.player.hp_black
            )
        features.current_health_ratio = health_ratio
        features.floor_stage = float(state.run.stage)

        for rule in deal.get("hard_constraints", []):
            feature = rule["feature"]
            op = rule.get("op", "==")
            value = float(rule.get("value", 0))
            current = getattr(features, feature, 0.0)
            if op == "==":
                passed = abs(current - value) < 1e-6
            elif op == ">=":
                passed = current >= value
            elif op == "<=":
                passed = current <= value
            elif op == ">":
                passed = current > value
            elif op == "<":
                passed = current < value
            else:
                passed = False
            checks.append(ConstraintCheck(feature, passed, current, op, value))

        allowed = all(c.passed for c in checks)
        return allowed, checks, features
