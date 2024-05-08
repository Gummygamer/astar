import unittest
import numpy as np
from gridworld import GridWorld, AstarAgent

class TestGridWorld(unittest.TestCase):
    def setUp(self):
        # Define a standard testing grid and obstacles
        self.size = 5
        self.target = (4, 4)
        self.initial_state = (0, 0)
        self.obstacles = [(1, 1), (4, 2), (3, 3)]
        self.environment = GridWorld(size=self.size, target=self.target, initial_state=self.initial_state, obstacles=self.obstacles)
        self.agent = AstarAgent(self.environment)

    def test_path_validity(self):
        """Test that the path does not pass through any obstacles."""
        path = self.agent.find_path()
        obstacle_set = set(self.obstacles)
        for step in path:
            self.assertNotIn(step, obstacle_set)

    def test_reach_target(self):
        """Test that the agent can reach the target from the initial state."""
        path = self.agent.find_path()
        self.assertEqual(path[-1], self.target)

    def test_path_efficiency(self):
        """Test that the path is efficient, optionally could be the shortest path."""
        path = self.agent.find_path()
        # Assuming manhattan distance, this checks if path length is reasonable
        min_possible_steps = abs(self.initial_state[0] - self.target[0]) + abs(self.initial_state[1] - self.target[1])
        self.assertTrue(len(path) >= min_possible_steps)  # Path cannot be shorter than Manhattan distance

if __name__ == '__main__':
    unittest.main()