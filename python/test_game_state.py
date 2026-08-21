import unittest

from game_state import GameState


class GameStateTests(unittest.TestCase):
    def test_from_dict_reads_pickups_collected(self):
        state = GameState.from_dict(
            {
                "episode_id": 3,
                "episode_tick": 17,
                "pickups_collected": 4,
                "terminal": True,
                "terminal_reason": "pickup_target_reached",
            }
        )

        self.assertEqual(state.episode_id, 3)
        self.assertEqual(state.episode_tick, 17)
        self.assertEqual(state.pickups_collected, 4)
        self.assertTrue(state.terminal)
        self.assertEqual(state.terminal_reason, "pickup_target_reached")

    def test_packed_grid_is_unpacked(self):
        packed = [
            "1,0,0|0,0,0",      # walls (2 rows x 3 cols)
            "0,0.25,1|0.5,0,0",
        ]
        state = GameState.from_dict({"grid": packed})
        self.assertEqual(state.grid, [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 0.25, 1.0], [0.5, 0.0, 0.0]],
        ])

    def test_legacy_nested_grid_still_accepted(self):
        nested = [[[0.0, 1.0], [1.0, 0.0]]]
        state = GameState.from_dict({"grid": nested})
        self.assertEqual(state.grid, nested)


if __name__ == "__main__":
    unittest.main()
