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
class DoorState:
    slot: int
    locked: bool = False
    open: bool = True
    target_room_index: int = -1
    position: tuple[float, float] | None = None

    @staticmethod
    def from_dict(data: dict) -> "DoorState":
        position = data.get("position")
        pos = None
        if isinstance(position, (list, tuple)) and len(position) >= 2:
            pos = (float(position[0]), float(position[1]))
        return DoorState(
            slot=int(data.get("slot", -1)),
            locked=bool(data.get("locked", False)),
            open=bool(data.get("open", True)),
            target_room_index=int(data.get("target_room_index", -1)),
            position=pos,
        )


@dataclass(frozen=True)
class PickupState:
    """Structured pickup record (coin/bomb/key/pill/card/...)."""
    variant: int = 0
    subtype: int = 0
    price: int = 0
    position: tuple[float, float] | None = None
    pos_x: float = 0.5
    pos_y: float = 0.5

    @staticmethod
    def from_dict(data: dict) -> "PickupState":
        position = data.get("position")
        pos = None
        if isinstance(position, (list, tuple)) and len(position) >= 2:
            pos = (float(position[0]), float(position[1]))
        return PickupState(
            variant=int(data.get("variant", 0)),
            subtype=int(data.get("subtype", 0)),
            price=int(data.get("price", 0)),
            position=pos,
            pos_x=float(data.get("pos_x", 0.5)),
            pos_y=float(data.get("pos_y", 0.5)),
        )


@dataclass(frozen=True)
class PedestalItemState:
    """Collectible on a pedestal; identity + price (negative = heart cost)."""
    item_id: int = 0
    price: int = 0
    position: tuple[float, float] | None = None
    pos_x: float = 0.5
    pos_y: float = 0.5

    @staticmethod
    def from_dict(data: dict) -> "PedestalItemState":
        position = data.get("position")
        pos = None
        if isinstance(position, (list, tuple)) and len(position) >= 2:
            pos = (float(position[0]), float(position[1]))
        return PedestalItemState(
            item_id=int(data.get("item_id", 0)),
            price=int(data.get("price", 0)),
            position=pos,
            pos_x=float(data.get("pos_x", 0.5)),
            pos_y=float(data.get("pos_y", 0.5)),
        )


@dataclass(frozen=True)
class SpecialRockState:
    """Identity-bearing rock variants (tinted rock / marked skull)."""
    kind: str = "tinted_rock"
    variant: int = 0
    position: tuple[float, float] | None = None

    @staticmethod
    def from_dict(data: dict) -> "SpecialRockState":
        position = data.get("position")
        pos = None
        if isinstance(position, (list, tuple)) and len(position) >= 2:
            pos = (float(position[0]), float(position[1]))
        return SpecialRockState(
            kind=str(data.get("kind", "tinted_rock")),
            variant=int(data.get("variant", 0)),
            position=pos,
        )


@dataclass(frozen=True)
class RunInfo:
    """Floor/room identity and route-relevant flags."""
    stage: int = 0
    stage_type: int = 0
    curse: int = 0
    room_type: int = 0
    room_shape: int = 0
    room_list_index: int = -1
    room_grid_index: int = -1
    room_clear: bool = False
    devil_deal_taken: bool = False
    angel_room_taken: bool = False

    @staticmethod
    def from_dict(data: dict) -> "RunInfo":
        return RunInfo(
            stage=int(data.get("stage", 0)),
            stage_type=int(data.get("stage_type", 0)),
            curse=int(data.get("curse", 0)),
            room_type=int(data.get("room_type", 0)),
            room_shape=int(data.get("room_shape", 0)),
            room_list_index=int(data.get("room_list_index", -1)),
            room_grid_index=int(data.get("room_grid_index", -1)),
            room_clear=bool(data.get("room_clear", False)),
            devil_deal_taken=bool(data.get("devil_deal_taken", False)),
            angel_room_taken=bool(data.get("angel_room_taken", False)),
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
    active_item_id: int = 0
    card_slot: int = 0
    pill_slot: int = 0
    trinket_id: int = 0
    hp_bone: float = 0.0
    hp_golden: float = 0.0
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
            active_item_id=int(data.get("active_item_id", 0)),
            card_slot=int(data.get("card_slot", 0)),
            pill_slot=int(data.get("pill_slot", 0)),
            trinket_id=int(data.get("trinket_id", 0)),
            hp_bone=float(data.get("hp_bone", 0.0)),
            hp_golden=float(data.get("hp_golden", 0.0)),
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
    pickups: list[PickupState] = field(default_factory=list)
    pedestal_items: list[PedestalItemState] = field(default_factory=list)
    special_rocks: list[SpecialRockState] = field(default_factory=list)
    doors: list[DoorState] = field(default_factory=list)
    run: RunInfo = field(default_factory=RunInfo)
    events: list[dict] = field(default_factory=list)
    raw: dict = field(default_factory=dict, repr=False)

    @staticmethod
    def from_dict(data: dict) -> "GameState":
        enemies = data.get("enemies", [])
        if not isinstance(enemies, list):
            enemies = []
        pickups = data.get("pickups", [])
        if not isinstance(pickups, list):
            pickups = []
        pedestal_items = data.get("pedestal_items", [])
        if not isinstance(pedestal_items, list):
            pedestal_items = []
        special_rocks = data.get("special_rocks", [])
        if not isinstance(special_rocks, list):
            special_rocks = []
        doors = data.get("doors", [])
        if not isinstance(doors, list):
            doors = []
        events = data.get("events", [])
        if not isinstance(events, list):
            events = []
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
            pickups=[PickupState.from_dict(p) for p in pickups if isinstance(p, dict)],
            pedestal_items=[PedestalItemState.from_dict(p) for p in pedestal_items if isinstance(p, dict)],
            special_rocks=[SpecialRockState.from_dict(r) for r in special_rocks if isinstance(r, dict)],
            doors=[DoorState.from_dict(d) for d in doors if isinstance(d, dict)],
            run=RunInfo.from_dict(data.get("run", {})),
            events=[e for e in events if isinstance(e, dict)],
            raw=data,
        )
