import numpy as np
import heapq

from occupancy_grid import OccupancyGrid
from scipy.ndimage import binary_dilation


class Planner:
    """Simple occupancy grid Planner"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])
        
    def set_occupancy_grid(self, occupancy_grid):
        self.grid = occupancy_grid
        
    def occupancy_grid_threshold(self, occupancy_grid, threshold=0.5):
        thresholded_grid = np.where(occupancy_grid > threshold, 1, 0)
        return thresholded_grid
    
    def occupancy_grid_dilate(self, occupancy_grid, radius=1):
        dilated_grid = binary_dilation(occupancy_grid, structure=np.ones((2*radius+1, 2*radius+1)))
        return dilated_grid
    
    def get_neighbours(self, current_cell):
        """
        Get the neighbours of a cell
        current_cell : [x, y] nparray, position of the current cell
        """
        x, y = current_cell
        neighbours = []

        # Check the 8 adjacent cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip the current cell
                neighbour = (x + dx, y + dy)  # Convert to tuple
                neighbours.append(neighbour)

        return neighbours

    def heuristic(self, cell_1, cell_2):
        """
        Compute the heuristic value between two cells
        cell 1 : [x, y] nparray, position of the first cell
        cell 2 : [x, y] nparray, position of the second cell
        """
        
        # Convert the inputs to numpy arrays
        cell_1 = np.array(cell_1)
        cell_2 = np.array(cell_2)

        # Calculate the Euclidean distance between the two cells
        distance = np.linalg.norm(cell_1 - cell_2)

        return distance

    def plan(self, start, goal):
        """
        A* path planning algorithm to find a path from start to goal
        start : [x, y] nparray, starting position
        goal : [x, y] nparray, goal position
        """
        
        start = self.grid.conv_world_to_map(start[0], start[1])
        goal = self.grid.conv_world_to_map(goal[0], goal[1])

        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start, goal), start))
        came_from = {}
        g_score = {tuple(start): 0}
        f_score = {tuple(start): self.heuristic(start, goal)}

        while open_set:
            current = heapq.heappop(open_set)[1]
            if np.array_equal(current, goal):
                return self.reconstruct_path(came_from, current)

            for neighbour in self.get_neighbours(current):
                current_np = np.array(current)
                neighbour_np = np.array(neighbour)

                tentative_g_score = g_score[tuple(current)] + np.linalg.norm(neighbour_np - current_np)
                if tuple(neighbour) not in g_score or tentative_g_score < g_score[tuple(neighbour)]:
                    came_from[tuple(neighbour)] = current
                    g_score[tuple(neighbour)] = tentative_g_score
                    f_score[tuple(neighbour)] = tentative_g_score + self.heuristic(neighbour, goal)
                    if tuple(neighbour) not in [i[1] for i in open_set]:
                        heapq.heappush(open_set, (f_score[tuple(neighbour)], neighbour))
   

        return None  # Return None if no path is found

    def reconstruct_path(self, came_from, current):
        total_path = [current]
        while tuple(current) in came_from:
            current = came_from[tuple(current)]
            total_path.append(current)
        return total_path[::-1]  # Return reversed path


    def explore_frontiers(self):
        """ Frontier based exploration """
        goal = np.array([0, 0, 0])  # frontier to reach for exploration
        # Will use TP1 wall follow instead
        return goal
