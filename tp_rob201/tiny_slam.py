""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np

from occupancy_grid import OccupancyGrid


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])
        
        self.refresh_counter = 0
        

    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # TODO for TP4
        # This function is left beeing reduntant as the masking already happens in the update_map function, but was left like this on purpose for the sake of the exercise and readability

        # Extract robot data
        # x_pose = pose[0]
        # y_pose = pose[1]
        # theta_pose = pose[2]
        
        lidar_values = lidar.get_sensor_values()
        lidar_angles = lidar.get_ray_angles()
        
        # get array of bool's where the lidar found obstacules
        in_range = lidar_values <= (lidar.max_range - 25)
        
        lidar_values = lidar_values[in_range]
        lidar_angles = lidar_angles[in_range]

        # Calculate moving average of lidar values
        lidar_values_ma = np.convolve(lidar_values, np.ones(5)/5, mode='same')

        # Check if the difference between the original value and the moving average is greater than a threshold
        diff = np.abs(lidar_values - lidar_values_ma)
        mask = diff < 0.01 * lidar_values
        
        # World coordinates in cartesian
        lidar_dx = pose[0] + lidar_values[mask] * np.cos(pose[2] + lidar_angles[mask])
        lidar_dy = pose[1] + lidar_values[mask] * np.sin(pose[2] + lidar_angles[mask])

        # get nearest grid point in the map and add its value to score
        x_px, y_px = self.grid.conv_world_to_map(lidar_dx, lidar_dy)
        select = np.logical_and(np.logical_and(x_px > 0, x_px < self.grid.x_max_map), np.logical_and(y_px > 0, y_px < self.grid.y_max_map))
        x_px = x_px[select]
        y_px = y_px[select]

        score = np.sum(self.grid.occupancy_map[x_px, y_px])

        return score

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4
        
        # if odom_pose_ref is not given, use the object's reference
        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref

        # odometer reference position
        x_odom_ref = odom_pose_ref[0]
        y_odom_ref = odom_pose_ref[1]
        angle_odom_ref = odom_pose_ref[2]

        # robot position in his odometer reference
        x_odom = odom_pose[0]
        y_odom = odom_pose[1]
        angle_odom = odom_pose[2]

        # distance travelled by the robot
        distance = np.sqrt(x_odom**2 + y_odom**2)

        # angle of the robot in the map frame
        ang_rotation = np.arctan2(y_odom, x_odom)

        # corrected position in the map frame
        x_corrected = x_odom_ref + distance * np.cos(ang_rotation + angle_odom_ref)
        y_corrected = y_odom_ref + distance * np.sin(ang_rotation + angle_odom_ref)

        corrected_pose = np.array([x_corrected, y_corrected, angle_odom + angle_odom_ref])

        return corrected_pose

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4

        # score with current reference
        best_pose_ref  = self.odom_pose_ref
        best_score = self._score(lidar, self.get_corrected_pose(raw_odom_pose,best_pose_ref))

        # randomly try to find better position close to the current one
        for i in range(150):
            delta = np.array([np.random.normal(0, 3, 1), np.random.normal(0, 3, 1), np.random.normal(0, 0.1, 1)])
            pose_i = self.get_corrected_pose(raw_odom_pose,best_pose_ref+delta.T[0])
            score = self._score(lidar, pose_i)
            i += 1

            # updates
            if score > best_score:
                best_score = score
                best_pose_ref = best_pose_ref + delta.T[0]
                i = 0

        # updates best position
        self.odom_pose_ref = best_pose_ref

        return best_score

    def update_map(self, lidar, pose):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        # TODO for TP3
        
        # Extract robot data
        x_pose = pose[0]
        y_pose = pose[1]
        theta_pose = pose[2]
        
        lidar_values = lidar.get_sensor_values()
        lidar_angles = lidar.get_ray_angles()
        
        log_odd_free = -0.10  # Log-odds value for free space
        log_odd_occ = 0.35    # Log-odds value for occupied space
        
        # get array of bool's where the lidar found obstacules
        border = 25
        in_range = lidar_values <= (lidar.max_range - border)
        
        lidar_values = lidar_values[in_range]
        lidar_angles = lidar_angles[in_range]

        # Calculate moving average of lidar values
        window_size = 5
        lidar_values_ma = np.convolve(lidar_values, np.ones(window_size)/window_size, mode='same')

        # Check if the difference between the original value and the moving average is greater than a threshold
        diff = np.abs(lidar_values - lidar_values_ma)
        mask = diff < 0.01 * lidar_values

        # Remove values that exceed the threshold
        lidar_values = lidar_values[mask]
        lidar_angles = lidar_angles[mask]
        
        # World coordinates in cartesian
        lidar_dx = x_pose + lidar_values * np.cos(theta_pose + lidar_angles)
        lidar_dy = y_pose + lidar_values * np.sin(theta_pose + lidar_angles)
        
        # Update free space along the line using Bresenham's algorithm every 5 points
        for x, y in zip(lidar_dx[::2], lidar_dy[::2]):
            self.grid.add_map_line(x_pose, y_pose, x, y, log_odd_free)
        
        # increase points values, obstacule
        self.grid.add_map_points(lidar_dx, lidar_dy, log_odd_occ)
        
        # Apply clamping to log-odds to avoid probability divergence
        self.grid.occupancy_map = np.clip(self.grid.occupancy_map, -4, 4)

        if self.refresh_counter % 10 == 0:
            self.grid.display_cv(pose)
            # self.grid.display_plt(pose)
            self.refresh_counter = 0
        else:
            self.refresh_counter += 1
            
            
            