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
    max_episode_steps: int = 3000
    action_timeout: float = 5.0
    n_workers: int = 1
    base_port: int = 9999



@dataclass
class RewardConfig:
    # Damage progress: scaled per HP point, capped cumulatively per episode
    # so total damage reward can never exceed damage_progress_cap.
    damage_dealt: float = 1.0
    damage_progress_cap: float = 0.25
    # Deprecated (redundant with damage progress): 0 disables it.
    enemy_killed: float = 0.0
    room_cleared: float = 2.0
    no_hit_bonus: float = 0.5
    # Penalty per half-heart lost.
    damage_taken: float = -0.5
    pickup_collected: float = 2.0
    wall_collision_penalty: float = 0.0
    # Per-use resource costs: each use must "pay for itself" through the
    # damage/clear/pickup rewards it enables, otherwise it nets negative.
    # This teaches resource economy without complex waste attribution.
    bomb_use_penalty: float = 0.0
    active_use_penalty: float = 0.0
    pillcard_use_penalty: float = 0.0
    death: float = -5.0
    timeout_penalty: float = -5.0
    # Small per-step cost: creates urgency without flipping strategy order.
    # Max drag over 3000 steps = -1.5, well below any positive outcome.
    time_penalty: float = -0.0005
    survival_bonus: float = 0.0
    floor_cleared: float = 100.0
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
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    total_timesteps: int = 1_000_000
    log_interval: int = 10
    save_interval: int = 50_000
    save_interval_seconds: float = 600.0
    eval_episodes: int = 20


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
