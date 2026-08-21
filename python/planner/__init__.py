"""Planner package: strategic layer of the hierarchical agent.

The planner consumes RunMemory + the knowledge base and emits Subgoals that
the goal-conditioned controller executes. See
docs/plans/hierarchical_agent_implementation.md for the roadmap.
"""

from .goals import GoalType, Subgoal  # noqa: F401
