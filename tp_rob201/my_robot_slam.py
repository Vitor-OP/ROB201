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
        elif self.counter < 500:
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
                new_occupancy_grid = self.planner.occupancy_grid_dilate(new_occupancy_grid, radius=1)
                self.planner.grid.occupancy_map = new_occupancy_grid
                
                # plt.imshow(new_occupancy_grid.T, origin='lower',
                #            extent=[self.occupancy_grid.x_min_world, self.occupancy_grid.x_max_world,
                #                    self.occupancy_grid.y_min_world, self.occupancy_grid.y_max_world])
                # plt.show()
                
                path = self.planner.plan(self.corrected_pose, np.array([0, 0, 0]))
                
                print(path)
                
                self.path_found = True
                
                self.counter += 1
                
                return potential_field_control(self.lidar(), self.corrected_pose, self.path[0])
                
            else:
                if self.path:
                    command = potential_field_control(self.lidar(), self.corrected_pose, self.path[0])
                    if np.linalg.norm(self.corrected_pose[:2] - self.path[0][:2]) < 15:
                        self.path.pop(0)
                    self.counter += 1
                    return command
                else:
                    #  case when self.path is empty
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
        
        goal = [-520, -480, 0]
        # goal = [-350, 35, 0]

        # Compute new command speed to perform obstacle avoidance
        command = potential_field_control(self.lidar(), pose, goal)

        return command
