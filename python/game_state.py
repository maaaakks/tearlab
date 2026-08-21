from dataclasses import dataclass, field


@dataclass(frozen=True)
class EnemyState:
    type: int
    variant: int
    hp: float
    position: tuple[float, float]
    init_seed: int = 0
    max_hp: float = 0.0
    velocity: tuple[float, float] = (0.0, 0.0)

    @staticmethod
    def from_dict(data: dict) -> "EnemyState":
        position = data.get("position", [0.0, 0.0])
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            position = [0.0, 0.0]
        velocity = data.get("velocity", [0.0, 0.0])
        if not isinstance(velocity, (list, tuple)) or len(velocity) < 2:
            velocity = [0.0, 0.0]
        return EnemyState(
            type=int(data.get("type", 0)),
            variant=int(data.get("variant", 0)),
            hp=float(data.get("hp", 0.0)),
            position=(float(position[0]), float(position[1])),
            init_seed=int(data.get("init_seed", 0)),
            max_hp=float(data.get("max_hp", 0.0)),
            velocity=(float(velocity[0]), float(velocity[1])),
        )


@dataclass(frozen=True)
class PlayerState:
    hp_red: float = 0.0
    hp_soul: float = 0.0
    hp_black: float = 0.0
    speed: float = 1.0
    damage: float = 3.5
    range: float = 6.5
    fire_rate: float = 10.0
    shot_speed: float = 1.0
    luck: float = 0.0
    num_bombs: int = 0
    num_keys: int = 0
    num_coins: int = 0
    has_active_item: bool = False
    active_charge: float = 0.0
    pos_x: float = 0.5
    pos_y: float = 0.5
    nearest_pickup_dx: float = 0.0
    nearest_pickup_dy: float = 0.0
    nearest_enemy_dx: float = 0.0
    nearest_enemy_dy: float = 0.0
    nearest_projectile_dx: float = 0.0
    nearest_projectile_dy: float = 0.0
    position: tuple[float, float] | None = None

    @staticmethod
    def from_dict(data: dict) -> "PlayerState":
        raw_pos = data.get("position")
        position: tuple[float, float] | None
        if isinstance(raw_pos, (list, tuple)) and len(raw_pos) >= 2:
            position = (float(raw_pos[0]), float(raw_pos[1]))
        else:
            position = None
        return PlayerState(
            hp_red=float(data.get("hp_red", 0.0)),
            hp_soul=float(data.get("hp_soul", 0.0)),
            hp_black=float(data.get("hp_black", 0.0)),
            speed=float(data.get("speed", 1.0)),
            damage=float(data.get("damage", 3.5)),
            range=float(data.get("range", 6.5)),
            fire_rate=float(data.get("fire_rate", 10.0)),
            shot_speed=float(data.get("shot_speed", 1.0)),
            luck=float(data.get("luck", 0.0)),
            num_bombs=int(data.get("num_bombs", 0)),
            num_keys=int(data.get("num_keys", 0)),
            num_coins=int(data.get("num_coins", 0)),
            has_active_item=bool(data.get("has_active_item", False)),
            active_charge=float(data.get("active_charge", 0.0)),
            pos_x=float(data.get("pos_x", 0.5)),
            pos_y=float(data.get("pos_y", 0.5)),
            nearest_pickup_dx=float(data.get("nearest_pickup_dx", 0.0)),
            nearest_pickup_dy=float(data.get("nearest_pickup_dy", 0.0)),
            nearest_enemy_dx=float(data.get("nearest_enemy_dx", 0.0)),
            nearest_enemy_dy=float(data.get("nearest_enemy_dy", 0.0)),
            nearest_projectile_dx=float(data.get("nearest_projectile_dx", 0.0)),
            nearest_projectile_dy=float(data.get("nearest_projectile_dy", 0.0)),
            position=position,
        )

    @property
    def total_hp(self) -> float:
        return self.hp_red + self.hp_soul + self.hp_black


@dataclass(frozen=True)
class GameState:
    episode_id: int = 0
    episode_tick: int = 0
    pickups_collected: int = 0
    terminal: bool = False
    terminal_reason: str | None = None
    room_cleared: bool = False
    player_dead: bool = False
    enemy_count: int = 0
    grid: list = field(default_factory=list)
    player: PlayerState = field(default_factory=PlayerState)
    enemies: list[EnemyState] = field(default_factory=list)
    raw: dict = field(default_factory=dict, repr=False)

    @staticmethod
    def from_dict(data: dict) -> "GameState":
        enemies = data.get("enemies", [])
        if not isinstance(enemies, list):
            enemies = []
        return GameState(
            episode_id=int(data.get("episode_id", 0)),
            episode_tick=int(data.get("episode_tick", 0)),
            pickups_collected=int(data.get("pickups_collected", 0)),
            terminal=bool(data.get("terminal", False)),
            terminal_reason=data.get("terminal_reason"),
            room_cleared=bool(data.get("room_cleared", False)),
            player_dead=bool(data.get("player_dead", False)),
            enemy_count=int(data.get("enemy_count", 0)),
            grid=data.get("grid", []),
            player=PlayerState.from_dict(data.get("player", {})),
            enemies=[EnemyState.from_dict(enemy) for enemy in enemies if isinstance(enemy, dict)],
            raw=data,
        )
