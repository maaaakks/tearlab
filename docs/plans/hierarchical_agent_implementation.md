# Hierarchical Agent — Implementation Roadmap

Status: active plan for branch `feature/hierarchical-agent`.
Design rationale: see `knowledge_guided_decision_making.md`.

## Architecture (3 layers, 3 algorithm classes)

```text
Strategic (~1/min)   route order, deals, shops, secret rooms
                     -> MCTS-style search over knowledge-base route graph
                        + learned leaf value fn (AlphaZero-lite)
Tactical (~1/sec)    active-item timing, fight vs flee, item pickup
                     -> small policy/value nets over entity embeddings
Control  (~10 Hz)    move, aim, dodge
                     -> existing live PPO controller (pretrained combat skill)
```

Key rule: deterministic game facts stay explicit (knowledge/*.json|yaml);
neural nets handle perception, control, and context-dependent valuation.

## Stage A — Interfaces (prerequisite for everything)

- [x] A1. Serialized state v2 (additive, non-breaking):
      stage/curse, room type/shape/index/stable ID, doors + destinations,
      pedestal items {id, price}, pickups {id, subtype}, gridEnt variants
      (tinted rock), inventory (coins/bombs/keys/hearts, active item,
      pocket items), deal history.
- [x] A2. Action expansion: MultiDiscrete([9, 9, bomb, active, card/pill])
      with validity masks (no bombs -> masked).
- [x] A3. RunMemory module: room graph + visited set updated per room
      transition; map summary encoding fed into controller obs.
- [ ] A4. Mod debug commands to inject controlled start states (spawn at
      Depths II with Polaroid + Fool) for rare route-segment training.

## Stage B — Knowledge base + planner

- [ ] B1. knowledge/items.json from the game's items.xml (ids, tags, quality).
- [ ] B2. Route DAGs (the_beast.yaml done; mega_satan.yaml next).
- [ ] B3. Rule-based planner: RunMemory + route DAG -> Subgoal stream.
      This alone yields a route-following autonomous agent before any new
      high-level learning.
- [ ] B4. Decision engine skeleton: hard constraints (rules) + utility
      features (learned later); deals.yaml schema done.

## Stage C — Learning the pieces

- [ ] C1. Vertical slice: tinted rock detect -> navigate -> bomb.
- [ ] C2. Entity-set transformer encoder for identity-bearing objects
      (items, prices, enemies) alongside the CNN grid.
- [ ] C3. Recurrent core (LSTM) in the controller for partial observability.
- [ ] C4. Behavior cloning of decision engine from scripted teacher.
- [ ] C5. Strategic search over route graph with learned value fn,
      trained on run outcomes.
- [ ] C6. Goal-conditioned fine-tuning per route segment, then full-floor
      curriculum; potential-based shaping along milestone DAG.

## Explicitly deferred

- World models / DreamerV3: knowledge base already provides most of what a
  learned model would give; revisit only if strategic search plateaus.

## Compatibility rules

- Observation changes must be additive; reuse the spatial combat encoder so
  existing checkpoints transfer (adapters / selective weight loading).
- Controller obs gains: goal vector (planner/goals.py) + map summary.
- Track decision accuracy and route progress separately from combat reward.
