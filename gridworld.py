import numpy as np
import time
from queue import PriorityQueue

class GridWorld:
    def __init__(self, size, target, initial_state, obstacles):
        self.size = size
        self.target = target
        self.initial_state = initial_state
        self.obstacles = set(obstacles)

    def is_obstacle(self, position):
        return position in self.obstacles

    def next_state(self, state, action):
        x, y = state
        next_positions = {
            'up': (max(x - 1, 0), y),
            'down': (min(x + 1, self.size - 1), y),
            'left': (x, max(y - 1, 0)),
            'right': (x, min(y + 1, self.size - 1))
        }
        next_state = next_positions[action]
        if self.is_obstacle(next_state):
            return state
        return next_state

    def get_neighbors(self, node):
        directions = ['up', 'down', 'left', 'right']
        neighbors = []
        for direction in directions:
            next_pos = self.next_state(node, direction)
            if next_pos != node and not self.is_obstacle(next_pos):
                neighbors.append((next_pos, 1))  # (position, cost)
        return neighbors

    def heuristic(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))
    
class AstarAgent:
    def __init__(self, model):
        self.model = model
        self.current_state = model.initial_state

    def display_grid(self):
        grid = np.array([[' ' for _ in range(self.model.size)] for _ in range(self.model.size)])
        for obs in self.model.obstacles:
            grid[obs[0]][obs[1]] = 'X'
        grid[self.current_state[0]][self.current_state[1]] = 'A'
        grid[self.model.target[0]][self.model.target[1]] = 'T'
        print("\n".join(["|".join(row) for row in grid]))
        print()

    def find_path(self):
        frontier = PriorityQueue()
        frontier.put((0, self.current_state))
        came_from = {}
        cost_so_far = {}
        came_from[self.current_state] = None
        cost_so_far[self.current_state] = 0

        while not frontier.empty():
            current = frontier.get()[1]

            if current == self.model.target:
                break

            for next, cost in self.model.get_neighbors(current):
                new_cost = cost_so_far[current] + cost
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.model.heuristic(next, self.model.target)
                    frontier.put((priority, next))
                    came_from[next] = current

        return self.reconstruct_path(came_from, self.model.target)

    def reconstruct_path(self, came_from, start):
        current = start
        path = []
        while current != self.current_state:
            path.append(current)
            current = came_from[current]
        path.reverse()
        return path

    def execute_plan(self):
        path = self.find_path()
        for state in path:
            self.current_state = state
            self.display_grid()
            time.sleep(1)

def simulate():
    environment = GridWorld(size=5, target=(4, 4), initial_state=(0, 0), obstacles=[(1, 1), (4, 2), (3, 3)])
    agent = AstarAgent(environment)
    agent.execute_plan()

simulate()

