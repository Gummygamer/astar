import numpy as np
import os
import time
from random import sample
from queue import PriorityQueue

class GridWorld:
    def __init__(self, size, target, initial_state, num_obstacles):
        self.size = size
        self.target = target
        self.initial_state = initial_state
        self.states = [(i, j) for i in range(size) for j in range(size)]
        
        # Remove the initial state and target from possible obstacle locations
        possible_locations = set(self.states) - {initial_state, target}
        
        # Convert set to list for sampling
        possible_locations_list = list(possible_locations)

        # Randomly select obstacle locations
        self.obstacles = set(sample(possible_locations_list, num_obstacles))

    def is_obstacle(self, position):
        return position in self.obstacles

    def get_neighbors(self, current):
        directions = [('up', (0, -1)), ('right', (1, 0)), ('down', (0, 1)), ('left', (-1, 0))]
        neighbors = []
        for direction, (dx, dy) in directions:
            nx, ny = current[0] + dx, current[1] + dy
            if 0 <= nx < self.size and 0 <= ny < self.size and (nx, ny) not in self.obstacles:
                neighbors.append((nx, ny))
        return neighbors

    def heuristic(self, a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    def display_grid(self, current_position):
        os.system('cls' if os.name == 'nt' else 'clear')
        grid = [[' ' for _ in range(self.size)] for _ in range(self.size)]
        for (ox, oy) in self.obstacles:
            grid[ox][oy] = 'X'
        tx, ty = self.target
        cx, cy = current_position
        grid[cx][cy] = 'A'
        grid[tx][ty] = 'T'
        for row in grid:
            print("|".join(row))
        print()

class ActiveInferenceAgent:
    def __init__(self, model):
        self.model = model
        self.current_position = model.initial_state

    def move_to_target(self):
        path = self.calculate_path()
        for position in path:
            self.current_position = position
            self.model.display_grid(self.current_position)
            time.sleep(1)

    def calculate_path(self):
        frontier = PriorityQueue()
        frontier.put((0, self.model.initial_state))
        came_from = {}
        cost_so_far = {}
        came_from[self.model.initial_state] = None
        cost_so_far[self.model.initial_state] = 0

        while not frontier.empty():
            current = frontier.get()[1]

            if current == self.model.target:
                break

            for next in self.model.get_neighbors(current):
                new_cost = cost_so_far[current] + 1  # Assume each move costs 1
                if next not in cost_so_far or new_cost < cost_so_far[next]:
                    cost_so_far[next] = new_cost
                    priority = new_cost + self.model.heuristic(next, self.model.target)
                    frontier.put((priority, next))
                    came_from[next] = current

        # Reconstruct path from came_from
        if current == self.model.target:  # Ensure a path was found
            path = []
            while current != self.model.initial_state:
                path.append(current)
                current = came_from[current]
            path.append(self.model.initial_state)
            path.reverse()
            return path
        return [self.model.initial_state]  # Return a minimal path if no path found

def simulate():
    environment = GridWorld(size=5, target=(4, 4), initial_state=(0, 0), num_obstacles=3)
    agent = ActiveInferenceAgent(environment)
    agent.move_to_target()

if __name__ == "__main__":
    simulate()
