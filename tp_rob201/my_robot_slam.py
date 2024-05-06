"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.entities.robot_abstract import RobotAbstract
from place_bot.entities.odometer import OdometerParams
from place_bot.entities.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid
from occupancy_grid import OccupancyGrid
from planner import Planner

from matplotlib import pyplot as plt

# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""

    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(should_display_lidar=False,
                         lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0
        
        self.path_found = False
        self.path = []
        self.paths = []

        # Init SLAM object
        self._size_area = (1200, 800)
        self.occupancy_grid = OccupancyGrid(x_min=- self._size_area[0],
                                            x_max=self._size_area[0],
                                            y_min=- self._size_area[1],
                                            y_max=self._size_area[1],
                                            resolution=2)
        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])

    def control(self):
        """
        Main control function executed at each time step
        """
        
        # return self.control_tp2()
                    
        if self.counter % 500 == 0:
            print(self.counter)
        
        # Begin without moving to create the base of the map
        if self.counter < 40:
            self.tiny_slam.update_map(self.lidar(),self.odometer_values())
            score = self.tiny_slam.localise(self.lidar(),self.odometer_values())
            
            self.counter += 1
            
            return {"forward": 0,
                    "rotation": 0}
        
        # Start moving and only update the map if the robot is localized
        elif self.counter < 6000:
            score = self.tiny_slam.localise(self.lidar(), self.odometer_values())
            if score > 65:
                self.corrected_pose = self.tiny_slam.get_corrected_pose(self.odometer_values())
                self.tiny_slam.update_map(self.lidar(), self.corrected_pose)
                
            self.counter += 1
                
            return self.control_tp1()
                
        else:
            score = self.tiny_slam.localise(self.lidar(), self.odometer_values())
            if score > 65 and self.path_found == False:
                self.corrected_pose = self.tiny_slam.get_corrected_pose(self.odometer_values())
                self.tiny_slam.update_map(self.lidar(), self.corrected_pose)
                
            if self.path_found == False:
                new_occupancy_grid = self.tiny_slam.grid.occupancy_map
                new_occupancy_grid = self.planner.occupancy_grid_threshold(new_occupancy_grid, threshold=-0.5)
                new_occupancy_grid = self.planner.occupancy_grid_dilate(new_occupancy_grid, radius=5)
                self.planner.grid.occupancy_map = new_occupancy_grid
                
                plt.imshow(new_occupancy_grid.T, origin='lower',
                           extent=[self.occupancy_grid.x_min_world, self.occupancy_grid.x_max_world,
                                   self.occupancy_grid.y_min_world, self.occupancy_grid.y_max_world])
                plt.show()
                
                path = self.planner.plan(self.corrected_pose, np.array([0, 0, 0]))
                
                # print(path)
                
                for i in path:
                    self.paths.append(self.planner.grid.conv_map_to_world(i[0], i[1]))
                    
                # print(self.paths)
                
                
                
                self.path_found = True
                
                self.counter += 1
                
                closest_point_index = np.argmin(np.linalg.norm(np.array(self.paths) - self.corrected_pose[:2], axis=1))
                if closest_point_index + 30 < len(self.paths):
                    goal_point = self.paths[closest_point_index + 30]
                    command = potential_field_control(self.lidar(), self.tiny_slam.get_corrected_pose(self.odometer_values()), goal_point)
                    return command
                else:
                    return {"forward": 0,
                            "rotation": 0}
        

            else:
                
                self.counter += 1
                
                closest_point_index = np.argmin(np.linalg.norm(np.array(self.paths) - self.corrected_pose[:2], axis=1))
                # print(closest_point_index)
                if closest_point_index + 30 < len(self.paths):
                    goal_point = self.paths[closest_point_index + 30]
                    command = potential_field_control(self.lidar(), self.tiny_slam.get_corrected_pose(self.odometer_values()), [goal_point[0], goal_point[1], 0])
                    return command
                else:
                    return {"forward": 0,
                            "rotation": 0}
        

    def control_tp1(self):
        """
        Control function for TP1
        """

        # Compute new command speed to perform obstacle avoidance
        command = reactive_obst_avoid(self.lidar())
        return command

    def control_tp2(self):
        """
        Control function for TP2
        """
        # pose = self.odometer_values()
        pose = self.tiny_slam.get_corrected_pose(self.odometer_values())
        
        # goal = [-520, -480, 0]
        goal = [-350, 35, 0]

        # Compute new command speed to perform obstacle avoidance
        command = potential_field_control(self.lidar(), pose, goal)

        return command
