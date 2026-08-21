import unittest

import numpy as np

from game_state import DoorState, GameState, RunInfo
from run_memory import (
    CHANNEL_CLEARED,
    CHANNEL_CURRENT,
    CHANNEL_ROOM_TYPE,
    CHANNEL_VISITED,
    NUM_MAP_CHANNELS,
    MAP_HEIGHT,
    MAP_WIDTH,
    RunMemory,
)


def make_state(
    list_index: int,
    grid_index: int,
    room_type: int = 1,
    stage: int = 1,
    clear: bool = True,
    doors: list[DoorState] | None = None,
) -> GameState:
    return GameState(
        run=RunInfo(
            stage=stage,
            stage_type=0,
            room_type=room_type,
            room_list_index=list_index,
            room_grid_index=grid_index,
            room_clear=clear,
        ),
        doors=doors or [],
    )


class RunMemoryTests(unittest.TestCase):
    def test_first_state_marks_current_room_only(self):
        mem = RunMemory()
        # grid_index 40 -> x=1, y=3
        state = make_state(list_index=10, grid_index=40, room_type=1)
        grid = mem.update(state)

        self.assertEqual(grid.shape, (NUM_MAP_CHANNELS, MAP_HEIGHT, MAP_WIDTH))
        self.assertEqual(grid.dtype, np.float32)
        self.assertEqual(grid[CHANNEL_CURRENT, 3, 1], 1.0)
        # Current room is not "visited" until we leave it
        self.assertEqual(grid[CHANNEL_VISITED, 3, 1], 0.0)
        self.assertAlmostEqual(grid[CHANNEL_ROOM_TYPE, 3, 1], 1 / 24.0)

    def test_room_transition_marks_previous_visited(self):
        mem = RunMemory()
        mem.update(make_state(list_index=10, grid_index=40))
        grid = mem.update(make_state(list_index=11, grid_index=41))

        # Previous room now visited
        self.assertEqual(grid[CHANNEL_VISITED, 3, 1], 1.0)
        # New current room
        self.assertEqual(grid[CHANNEL_CURRENT, 3, 2], 1.0)
        self.assertEqual(int(grid[CHANNEL_CURRENT].sum()), 1)

    def test_door_edges_recorded_bidirectionally(self):
        mem = RunMemory()
        doors = [DoorState(slot=3, target_room_index=11)]  # RIGHT
        mem.update(make_state(list_index=10, grid_index=40, doors=doors))
        mem.update(make_state(list_index=11, grid_index=41))

        self.assertEqual(mem.adjacency[10][3], 11)
        self.assertEqual(mem.adjacency[11][2], 10)  # LEFT back-edge inferred

    def test_floor_change_resets_memory(self):
        mem = RunMemory()
        mem.update(make_state(list_index=10, grid_index=40, stage=1))
        grid = mem.update(make_state(list_index=50, grid_index=91, stage=2))

        # Old floor's rooms must be gone; only the new current room remains
        self.assertEqual(int(grid[CHANNEL_VISITED].sum()), 0)
        x, y = RunMemory._grid_xy(91)  # 91 % 13 = 0, 91 // 13 = 7
        self.assertEqual(grid[CHANNEL_CURRENT, y, x], 1.0)
        self.assertEqual(len(mem.rooms), 1)

    def test_cleared_flag_tracked(self):
        mem = RunMemory()
        mem.update(make_state(list_index=10, grid_index=40, clear=True))
        mem.update(make_state(list_index=11, grid_index=41, clear=False))
        grid = mem.map_summary()
        self.assertEqual(grid[CHANNEL_CLEARED, 3, 1], 1.0)
        self.assertEqual(grid[CHANNEL_CLEARED, 3, 2], 0.0)


if __name__ == "__main__":
    unittest.main()
