# Knowledge-Guided Decision Making

Status: design note, not implemented.

## Objective

Build an Isaac agent that does more than react locally. The agent should combine
learned movement and combat skills with explicit game knowledge so it can make
necessary long-horizon decisions: choose a route, manage resources, value
items, recognize special objects, and satisfy route-specific prerequisites.

The target is not a single hard-coded walkthrough. The target is a
goal-conditioned agent that can pursue objectives such as `THE_BEAST` or
`MEGA_SATAN` while adapting decisions to the current run.

## Current Baseline

Phase 1A is deliberately an aggressive combat bootcamp:

- Isaac restarts in a normal starting-room layout.
- The mod injects exactly four Gapers.
- Enemy positions are randomized within a configured radius.
- No additional obstacles are injected by Phase 1A.
- The episode ends on death, timeout, or room clear.

This is useful for learning movement, aiming, dodging, and killing under
pressure. It is not yet a natural run distribution: the agent does not traverse
a floor, retain a map, choose a route, shop, or make item decisions.

The current observation contains a local spatial grid and player statistics.
It represents walls, generic obstacles, pits, the player, enemies,
projectiles, generic pickups, and doors. It does not currently expose enough
information for strategic play:

- no stage, floor, room type, or room identity;
- no persistent map or visited-room history;
- no pedestal item identity or item quality;
- no pickup or deal price;
- no detailed inventory of item IDs, trinkets, cards, and pills;
- no distinction between ordinary and tinted rocks;
- no secret-room candidates;
- no Angel/Devil decision state or route prerequisites;
- no long-term memory in the feed-forward policy.

The current action space also only covers movement and shooting. Bombs, active
items, cards, pills, and other interactions must be added before the agent can
execute advanced routes.

## Core Design Decision: Use a Hybrid Agent

Pure end-to-end PPO should not be expected to rediscover every Isaac rule from
a reward delivered at the end of a long run. The signal would be extremely
sparse, and many important prerequisites would be encountered too rarely.

Use a hierarchical hybrid architecture instead:

```text
Run objective (THE_BEAST, MEGA_SATAN, SURVIVE, ...)
                         |
                         v
Strategic planner + explicit game knowledge
                         |
                  current subgoal
                         |
                         v
Goal-conditioned learned controller
        (combat, dodging, navigation, interaction)
                         |
                         v
                       Isaac
```

Responsibilities should be separated:

1. **Strategic planner**
   Tracks route prerequisites and chooses the next subgoal.

2. **Decision engine**
   Evaluates items, deals, shops, resources, and conditional risks using both
   explicit rules and learned utility estimates.

3. **Learned controller**
   Executes the selected subgoal through movement, shooting, dodging, and
   interactions.

4. **Run memory**
   Stores the map, visited rooms, inventory, route state, and past decisions.
   This can start as explicit structured memory and later be complemented by a
   recurrent network.

Stable, deterministic game facts should remain explicit and inspectable. The
neural policy should focus on uncertain perception, control, combat, and
context-dependent utility rather than memorizing rules that are already known.

## Required Observation Expansion

| Decision | Required state |
| --- | --- |
| Floor routing | Stage, floor, room index, room type, doors, visited map |
| Shops | Item ID, price, coins, restock state, affordability |
| Devil Deals | Item ID, heart price, current health, deal history |
| Angel Rooms | Room type, Angel chance, deal history, route objective |
| Tinted rocks | A distinct grid type or object record plus bomb count |
| Secret rooms | Persistent room graph and rule-generated candidate walls |
| Item choice | Item IDs, tags, quality, build, character, health, route |
| Angel statues | Statue state, Key Pieces owned, bomb count, route objective |
| The Beast route | Route stage, photo/card state, Strange Door and Dad's Note |

Prefer structured object records where identity matters. A generic pickup
channel is not sufficient for items, prices, cards, deals, statues, or special
rocks.

Suggested run-level fields include:

- target route and current route stage;
- stage and floor variant;
- current room type and stable room identifier;
- visited-room graph and known doors;
- keys, bombs, coins, all health types, and consumables;
- passive item IDs, active item, trinket, card, and pill;
- Angel/Devil eligibility and deal history;
- Key Pieces and other quest items;
- nearby interactable objects with type, subtype, price, and position;
- planner-selected target type and target position.

## Required Action Expansion

Add explicit branches or commands for:

- movement;
- shooting;
- placing a bomb;
- using an active item;
- using a card or pill;
- dropping or swapping an item when necessary.

Impossible actions should be masked. For example, bomb use should be masked
when no bombs are available. Action masking prevents the policy from wasting
training samples on mechanically invalid choices; it must not silently encode
all strategic decisions.

## Example: The Beast Route

For a `THE_BEAST` objective, the planner should decompose the run into explicit
subgoals rather than reward only the final victory:

```text
Reach Depths II
-> locate the marked skull
-> preserve and place a bomb
-> collect The Fool
-> defeat Mom
-> take The Polaroid or The Negative
-> use The Fool to leave the boss room
-> open the Strange Door
-> reach and collect Dad's Note
-> complete the Ascent
-> reach Home
-> defeat Dogma and The Beast
```

Each transition needs a clear success condition, a bounded reward, and a way to
start training near that situation. Otherwise the later stages will be too rare
to learn.

## Goal-Conditioned Decisions

Rules must depend on the selected objective and the current run.

### Angel Rooms and Devil Deals

"Never take a Devil Deal" is not a universal rule.

- For a `MEGA_SATAN` route, preserving Angel access and obtaining both Key
  Pieces can be a hard strategic priority.
- For a `THE_BEAST` route, Angel Rooms are not a mandatory prerequisite. A
  Devil Deal may be worthwhile if its expected survival value exceeds its
  health and opportunity cost.
- A deal must also be evaluated against the current character, health, build,
  route, and remaining floor requirements.

The decision engine should expose values such as `affordable`,
`route_compatible`, `survival_utility`, and `opportunity_cost` instead of a
single unconditional yes/no rule.

### Shops

The agent needs the exact price, available coins, item identity, current build,
and future resource needs. Affordability is deterministic; purchase utility is
context-dependent.

### Tinted Rocks and Secret Rooms

Tinted rocks should be exposed directly as a distinct game-state object. Secret
rooms are better handled by a rule-based candidate generator operating on the
remembered room graph. The learned controller can then navigate to and bomb the
selected candidate.

### Conditional Item Avoidance

Item value depends on character, inventory, synergies, health, route, and the
agent's ability to use the item safely. Store item facts and tags in a knowledge
base, then combine them with current-run features. Avoid forcing PPO to infer
thousands of stable item descriptions from rare outcomes.

### Angel Statues and Key Pieces

For a compatible route, the planner should recognize the statue, verify bomb
availability and existing Key Pieces, select the statue as a target, and then
ask the learned controller to execute the bomb-and-combat sequence.

## Knowledge Representation

Store explicit knowledge in versioned, machine-readable files, for example:

```text
knowledge/
├── items.json
├── entities.json
├── prices.json
├── room_rules.json
├── routes/
│   ├── the_beast.yaml
│   └── mega_satan.yaml
└── decision_rules/
    ├── deals.yaml
    ├── shops.yaml
    └── resources.yaml
```

This knowledge should be compiled into numeric features, target proposals,
utility labels, and route state. Natural-language text should not be queried in
the real-time control loop. An LLM may help author or audit rules offline, but
runtime behavior should remain deterministic, fast, and testable.

## Introducing Knowledge Before, During, and After RL

### Before reinforcement learning

- Create the structured knowledge base and route definitions.
- Build a scripted teacher that can select valid strategic subgoals.
- Generate demonstrations for deals, shops, bombs, route transitions, and item
  choices.
- Pretrain the high-level decision model with behavior cloning.
- Pretrain auxiliary predictions such as room type, affordability, object type,
  and route relevance.

### During reinforcement learning

- Pass the selected subgoal into the policy as a goal vector.
- Use dense, bounded rewards for progress toward the current subgoal.
- Reward completion of route milestones, not only final victory.
- Use action masks for invalid actions.
- Mix teacher demonstrations with self-play experience.
- Gradually reduce teacher intervention instead of removing it immediately.
- Replay rare strategic situations from controlled starts.
- Track decision accuracy and route progress separately from combat reward.

Potential-based shaping should be preferred where possible so intermediate
rewards guide progress without changing the intended optimal solution.

### After initial training

- Fine-tune on natural, complete runs with less shaping.
- Evaluate on fixed seeds and unseen seeds.
- Collect failure cases and turn them into targeted scenarios.
- Compare the agent's decisions with the teacher, then review disagreements
  where the learned policy may have found a better context-dependent choice.
- Reduce hard-coded utility rules only when learned replacements outperform
  them reliably.

## Proposed Curriculum

1. **Combat bootcamp**
   Current Phase 1A: movement, aiming, dodging, and clearing four Gapers.

2. **Interaction skills**
   Bomb a marked target, collect a specific pickup, use an active item, and use
   a card.

3. **Room navigation**
   Clear natural layouts, select a door, revisit rooms, and follow a target on
   a remembered map.

4. **Resource decisions**
   Learn affordability, shop choices, health costs, bomb/key conservation, and
   conditional item utility.

5. **Route segments**
   Train individual Beast, Mega Satan, and other route transitions from
   controlled starting states.

6. **Floor-scale play**
   Join room navigation, combat, resources, and boss progression.

7. **Complete natural runs**
   Fine-tune the hierarchical agent with the teacher mostly disabled.

The current Phase 3/4 configuration files express an intention to use natural
spawns, but the runtime still terminates an episode when a combat room is
cleared. Full-floor lifecycle, floor-change detection, and real floor rewards
must be implemented before those phases represent complete runs.

## Reusing the Current Combat Agent

Changing observation or action shapes makes existing full checkpoints
architecturally incompatible. Preserve the value of current training by making
the combat controller a module:

- retain or transfer the spatial combat encoder;
- keep movement and shooting heads reusable;
- add separate interaction heads and strategic inputs;
- place the planner above the low-level controller;
- use adapters or selective weight loading when observation fields expand.

The current checkpoint should become a pretrained combat skill, not the entire
future full-run brain.

## Evaluation

Track separate metrics for each capability:

- room-clear rate, survival time, damage taken, and combat efficiency;
- navigation success and steps required to reach a target;
- valid interaction rate and wasted-resource rate;
- shop/deal decision agreement with the teacher;
- tinted-rock and secret-room opportunity success;
- item-choice utility and avoidable bad-item rate;
- completion rate for each route milestone;
- full-route completion rate by fixed and unseen seed;
- dependence on teacher intervention over time.

A single average reward is not enough to determine whether strategic knowledge
has actually been learned.

## Recommended Implementation Order

1. Version a richer serialized run state without breaking combat training.
2. Add bomb, active-item, and card/pill actions with validity masks.
3. Add persistent room-graph and inventory memory.
4. Define a goal/subgoal interface between planner and controller.
5. Implement one narrow vertical slice: detect a tinted rock, preserve a bomb,
   navigate to it, and bomb it.
6. Add the machine-readable knowledge base and unit-tested decision engine.
7. Add demonstrations and behavior-cloning support.
8. Train route segments before attempting complete Beast runs.
9. Integrate full-floor lifecycle and natural-run evaluation.

## Open Questions for Later

- Which decisions must remain hard constraints, and which should be learned?
- Should the high-level policy be rule-based, learned, or a mixture of both?
- How should controlled game states be created for rare route segments?
- Which item knowledge should be exact, tagged, or learned from outcomes?
- How much game-state information is acceptable versus pixel-only perception?
- Should run memory remain explicit, recurrent, or combine both approaches?
- How should knowledge and route definitions be versioned across game updates?

