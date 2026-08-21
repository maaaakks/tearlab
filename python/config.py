import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class EnvConfig:
    host: str = "127.0.0.1"
    port: int = 9999
    grid_width: int = 13
    grid_height: int = 7
    grid_channels: int = 10
    player_features: int = 22
    normalize_player_features: bool = True
    mask_distance_vectors: bool = False
    frame_skip: int = 3
    frame_stack: int = 1
    free_running: bool = True
    enable_map_memory: bool = False
    # TODO D5: append the planner's goal vector and the decision engine's
    # numeric features as fixed observation vectors. Additive keys — legacy
    # checkpoints without them keep their state_dict (see C6).
    enable_goal_conditioning: bool = False
    # full_run: keep the episode alive across room clears so the agent learns
    # whole runs (combat + navigation + loot in the same rollout). When false,
    # clearing a room ends the episode (mono-room test/benchmark mode).
    full_run: bool = False
    max_episode_steps: int = 3000
    action_timeout: float = 5.0
    n_workers: int = 1
    base_port: int = 9999



@dataclass
class RewardConfig:
    # Damage progress: reward for clearing 100% of a room's enemy HP, scaled
    # by the fraction of room progress represented by each point of damage.
    # damage_progress_cap bounds the total damage reward per ROOM, so reward
    # is comparable across rooms regardless of enemy count or total HP.
    damage_dealt: float = 1.0
    damage_progress_cap: float = 0.25
    # Deprecated (redundant with damage progress): 0 disables it.
    enemy_killed: float = 0.0
    room_cleared: float = 2.0
    no_hit_bonus: float = 0.5
    # Speed is rewarded only after a successful room clear. The bonus decays
    # linearly to zero over clear_speed_horizon room steps, so surviving longer
    # is never punished and a fast death can never benefit from urgency.
    clear_speed_bonus: float = 0.0
    clear_speed_horizon: int = 0
    # Penalty per half-heart lost.
    damage_taken: float = -0.5
    # Generic pickup reward is OFF by default: pickup value depends on price,
    # need, build and route (see D4/D5 decision engine). Legacy nav phases
    # opt back in explicitly.
    pickup_collected: float = 0.0
    wall_collision_penalty: float = 0.0
    # Per-use resource costs: each use must "pay for itself" through the
    # damage/clear/pickup rewards it enables, otherwise it nets negative.
    # This teaches resource economy without complex waste attribution.
    bomb_use_penalty: float = 0.0
    active_use_penalty: float = 0.0
    pillcard_use_penalty: float = 0.0
    # Aim discipline: firing is free in Isaac, so unshaped policies spray.
    # Penalize shots that are misaligned with the nearest enemy (linear in
    # angular error beyond the tolerance cone) and shots fired with no
    # enemy on screen. Zero disables.
    aim_misalignment_penalty: float = 0.0   # max cost per fully-misaligned shot step
    aim_tolerance_deg: float = 30.0         # error below this counts as aimed
    aim_shoot_no_enemy_penalty: float = 0.0 # flat cost per shot step with no enemies
    death: float = -5.0
    # Extra negative terminal cost that decays from its full value at the
    # beginning of an episode to zero at the episode horizon. Keep negative.
    death_early_penalty: float = 0.0
    timeout_penalty: float = -5.0
    # Additional timeout cost multiplied by the fraction of enemy HP left in
    # the current room. Keep negative; zero preserves legacy behavior.
    timeout_enemy_hp_penalty: float = 0.0
    # Small per-step cost: creates urgency without flipping strategy order.
    # Max drag over 3000 steps = -1.5, well below any positive outcome.
    time_penalty: float = -0.0005
    survival_bonus: float = 0.0
    # Run-level milestones (all bounded one-shot rewards).
    boss_killed: float = 0.5
    floor_cleared: float = 2.0
    run_won: float = 10.0
    # One-shot exploration bonus for the first visit of a room this run.
    room_first_visit: float = 0.05
    # Deal protection signal (Angel/Devil preservation), floor-scoped:
    # penalty per half-heart of RED heart damage, plus a one-shot bonus when
    # a floor is cleared with red hearts intact.
    deal_protected_penalty: float = 0.0
    deal_protected_floor_bonus: float = 0.0
    pickup_approach_scale: float = 0.0
    nav_progress_scale: float = 0.0
    nav_reach_bonus: float = 0.0
    nav_reach_radius: float = 20.0
    nav_target_x: float | None = None
    nav_target_y: float | None = None
    nav_target_dx: float = 0.0
    nav_target_dy: float = 0.0


@dataclass
class TrainConfig:
    algorithm: str = "PPO"
    live_mode: bool = True
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    target_kl: float | None = None
    ent_coef: float = 0.01
    ent_coef_final: float | None = None
    entropy_decay_fraction: float = 1.0
    # Single-policy auxiliary objective for the shooting action head. It uses
    # the nearest-enemy vector already present in each observation: direction
    # labels when an enemy is visible, idle otherwise. The coefficient follows
    # the same linear schedule pattern as entropy.
    aim_aux_coef: float = 0.0
    aim_aux_final: float | None = None
    aim_aux_decay_fraction: float = 1.0
    aim_aux_idle_weight: float = 0.25
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 1_000_000
    log_interval: int = 10
    save_interval: int = 50_000
    save_interval_seconds: float = 600.0
    eval_episodes: int = 20
    # C4: recurrent memory (GRU) over partially-observable runs. When enabled,
    # each transition stores its per-step hidden state so async collection and
    # PPO updates stay consistent across batch boundaries.
    recurrent: bool = False
    recurrent_hidden_size: int = 128
    # C5: auxiliary representation-learning heads (room type, danger, next
    # destination, reachability). 0 disables them (legacy behavior).
    aux_coef: float = 0.0
    # E9: normalize the return targets of each PPO batch with running stats.
    # OFF by default (legacy behavior); turn on if value loss destabilizes on
    # long full runs where returns span many reward magnitudes.
    normalize_returns: bool = False
    return_norm_decay: float = 0.999
    # E6: fraction of episode resets that are controlled (rare-scenario) starts,
    # mixed into natural runs. E7: the fraction decays linearly to
    # controlled_reset_final over controlled_reset_decay_fraction of training.
    controlled_reset_fraction: float = 0.0
    controlled_reset_final: float = 0.0
    controlled_reset_decay_fraction: float = 1.0
    controlled_scenarios: list = field(default_factory=list)


@dataclass
class PhaseConfig:
    enemy_type: int = 10
    enemy_variant: int = 0
    enemy_count: int = 1
    enemy_collision_damage: float | None = None
    spawn_pickup_penny: bool = False
    pickup_random_position: bool = False
    pickup_offset_x: float = 180.0
    pickup_offset_y: float = 0.0
    pickup_radius_min: float = 120.0
    pickup_radius_max: float = 200.0
    terminal_on_pickup: bool = False
    terminal_pickup_count: int = 0
    respawn_pickup: bool = False
    spawn_enemies: bool = True
    random_spawn_positions: bool = False
    spawn_radius_min: float = 80.0
    spawn_radius_max: float = 160.0
    disable_shooting: bool = False
    mask_shoot: bool = False
    enable_interactions: bool = False
    spawn_obstacles: bool = False
    obstacle_count: int = 0
    obstacle_type: int = 4
    obstacle_min_spacing: int = 2
    # Strategic objective + resource preservation hints consumed by the rule
    # planner when env.enable_goal_conditioning is on (D5).
    objective: str = "THE_BEAST"
    preserve_angel: bool = False
    preserve_bombs: bool = False


@dataclass
class WandbConfig:
    enabled: bool = False
    project: str = "binding-of-ai"
    entity: Optional[str] = None
    run_name: Optional[str] = None
    tags: list = field(default_factory=list)


@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    phase: PhaseConfig = field(default_factory=PhaseConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def to_game_settings(self) -> dict:
        """Build the settings dict sent to the Lua mod's configure command."""
        return {
            "enemy_type": self.phase.enemy_type,
            "enemy_variant": self.phase.enemy_variant,
            "enemy_count": self.phase.enemy_count,
            "enemy_collision_damage": self.phase.enemy_collision_damage,
            "spawn_pickup_penny": self.phase.spawn_pickup_penny,
            "pickup_random_position": self.phase.pickup_random_position,
            "pickup_offset_x": self.phase.pickup_offset_x,
            "pickup_offset_y": self.phase.pickup_offset_y,
            "pickup_radius_min": self.phase.pickup_radius_min,
            "pickup_radius_max": self.phase.pickup_radius_max,
            "terminal_on_pickup": self.phase.terminal_on_pickup,
            "terminal_pickup_count": self.phase.terminal_pickup_count,
            "respawn_pickup": self.phase.respawn_pickup,
            "spawn_enemies": self.phase.spawn_enemies,
            "random_spawn_positions": self.phase.random_spawn_positions,
            "spawn_radius_min": self.phase.spawn_radius_min,
            "spawn_radius_max": self.phase.spawn_radius_max,
            "disable_shooting": self.phase.disable_shooting,
            "spawn_obstacles": self.phase.spawn_obstacles,
            "obstacle_count": self.phase.obstacle_count,
            "obstacle_type": self.phase.obstacle_type,
            "obstacle_min_spacing": self.phase.obstacle_min_spacing,
            "frame_skip": self.env.frame_skip,
            "max_episode_ticks": self.env.max_episode_steps,
            "full_run": self.env.full_run,
        }


def config_from_dict(data: dict | None = None) -> Config:
    """Build a config from already-loaded data, falling back to defaults."""
    config = Config()
    if not data:
        return config

    for section_name, section_cls in [
        ("env", EnvConfig),
        ("reward", RewardConfig),
        ("train", TrainConfig),
        ("phase", PhaseConfig),
        ("wandb", WandbConfig),
    ]:
        if section_name in data:
            section = getattr(config, section_name)
            for key, value in data[section_name].items():
                if hasattr(section, key):
                    setattr(section, key, value)

    return config


def load_config(path: str | None = None) -> Config:
    """Load config from YAML file, falling back to defaults."""
    if path is None:
        return Config()

    with open(path) as f:
        data = yaml.safe_load(f)
    return config_from_dict(data)
