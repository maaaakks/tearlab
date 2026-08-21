"""RunMemory: persistent per-floor map memory for the hierarchical agent.

Tracks the room graph of the current floor from serialized state v2 fields
(run.room_list_index, run.room_grid_index, run.room_type, doors). Produces a
compact spatial "map" observation channel stack that lets the controller know
where it has been, where it is, and what kind of rooms are known.

Isaac floors are laid out on a 13x9 grid; a room's GridIndex encodes
(x, y) as x + 13*y. All channels are float32 in [0, 1].
"""

import numpy as np

MAP_WIDTH = 13
MAP_HEIGHT = 9

CHANNEL_VISITED = 0    # 1.0 for any room seen this floor
CHANNEL_CURRENT = 1    # 1.0 for the room the player is in
CHANNEL_ROOM_TYPE = 2  # room_type / ROOM_TYPE_NORM, clipped to [0, 1]
CHANNEL_CLEARED = 3    # 1.0 for rooms last seen cleared
NUM_MAP_CHANNELS = 4

ROOM_TYPE_NORM = 24.0  # max RoomType enum value (~26); conservative divisor


class RunMemory:
    """Room graph + visited-set memory for one floor."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self.floor_key: tuple[int, int] | None = None
        self.rooms: dict[int, dict] = {}   # list_index -> room record
        self.adjacency: dict[int, dict[int, int]] = {}  # list_index -> {slot: target}
        self.current_list_index: int | None = None

    @staticmethod
    def _grid_xy(grid_index: int) -> tuple[int, int]:
        x = grid_index % MAP_WIDTH
        y = (grid_index // MAP_WIDTH) % MAP_HEIGHT
        return x, y

    def _room_pos(self, record: dict) -> tuple[int, int]:
        gi = record.get("grid_index", -1)
        if gi is None or gi < 0:
            return (MAP_WIDTH // 2, MAP_HEIGHT // 2)
        return self._grid_xy(gi)

    def update(self, state) -> np.ndarray:
        """Fold a GameState into memory and return the map summary tensor."""
        run = state.run
        floor_key = (run.stage, run.stage_type)
        if floor_key != self.floor_key:
            # New floor (or first state): wipe memory.
            self.reset()
            self.floor_key = floor_key

        li = run.room_list_index
        if li is None or li < 0:
            li = run.room_grid_index

        if self.current_list_index is not None and li != self.current_list_index:
            prev = self.rooms.get(self.current_list_index)
            if prev is not None:
                prev["visited"] = True

        record = self.rooms.get(li)
        if record is None:
            record = {
                "grid_index": run.room_grid_index,
                "room_type": run.room_type,
                "visited": self.current_list_index is not None,
                "clear_last_seen": bool(run.room_clear),
                "first_visit": True,
            }
            self.rooms[li] = record
        else:
            record["grid_index"] = run.room_grid_index
            record["room_type"] = run.room_type
            record["clear_last_seen"] = bool(run.room_clear)
            record["visited"] = True
            record["first_visit"] = False

        # Adjacency from this room's doors (both directions where possible)
        edges = self.adjacency.setdefault(li, {})
        for door in getattr(state, "doors", []) or []:
            target = door.target_room_index
            if target is None or target < 0:
                continue
            edges[door.slot] = target
            self.adjacency.setdefault(target, {})[_OPPOSITE_SLOT.get(door.slot, door.slot)] = li

        self.current_list_index = li
        return self.map_summary()

    def map_summary(self) -> np.ndarray:
        """(NUM_MAP_CHANNELS, MAP_HEIGHT, MAP_WIDTH) float32 in [0, 1]."""
        grid = np.zeros((NUM_MAP_CHANNELS, MAP_HEIGHT, MAP_WIDTH), dtype=np.float32)
        for li, record in self.rooms.items():
            x, y = self._room_pos(record)
            if record.get("visited"):
                grid[CHANNEL_VISITED, y, x] = 1.0
            if record.get("clear_last_seen"):
                grid[CHANNEL_CLEARED, y, x] = 1.0
            grid[CHANNEL_ROOM_TYPE, y, x] = min(
                1.0, (record.get("room_type") or 0) / ROOM_TYPE_NORM
            )
        if self.current_list_index in self.rooms:
            x, y = self._room_pos(self.rooms[self.current_list_index])
            grid[CHANNEL_CURRENT, y, x] = 1.0
        return grid


# Door slots: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4..7 special
_OPPOSITE_SLOT = {
    0: 1,
    1: 0,
    2: 3,
    3: 2,
}
