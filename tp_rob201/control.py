""" A set of robotics control functions """

import random
import numpy as np

def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    
    # TODO for TP1
    # I did went a bit overkill, where where asked a simple obstacle avoidance with random rotation,
    # I instead implemented a "follow the wall" algorithm, where the robot will try to follow the wall on its left, without collinding with it.
    
    lidar_values = lidar.get_sensor_values()
    lidar_angles = lidar.get_ray_angles()
    lidar_len = len(lidar_values)
    
    # INDEX:
    # Lidar values are right, middle, left
    # positive rotation is left, negative is right
    
    middle_index = lidar_len // 2 # middle index is the front of the robot
    side_index = lidar_len // 4 # side index is the right of the robot (and middle + side is the left of the robot)
    
    # Frontal collision avoidance (low range, big field of view)
    if np.any(lidar_values[middle_index - int(lidar_len*0.1):middle_index + int(lidar_len*0.1)] < 40):
        speed = -0.1
        rotation_speed = -0.4
    
    # Far from any wall, just go forward max speed (made taking into account the start of the simulation)
    elif np.all(lidar_values[:] > 80):
        speed = 1
        rotation_speed = 0
    
    # Foward going (no obstacle in front)
    else:
        # Slighltly turn right if going forward but close to a wall on the left
        if np.any(lidar_values[middle_index + int(lidar_len*0.1):middle_index + int(lidar_len*0.15)]) < 75: 
            speed = 0.3
            rotation_speed = 0.2
        
        else: # no wall in left side
                speed = -0.3
                rotation_speed = -0.75

    command = {"forward": speed,
               "rotation": rotation_speed}

    return command


def potential_field_control(lidar, current_pose, goal_pose):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """
    # TODO for TP2
    
    lidar_values = lidar.get_sensor_values()
    lidar_angles = lidar.get_ray_angles()
    lidar_len = len(lidar_values)

    # Define the potential field
    potential_field = np.zeros_like(lidar_values)

    # Compute the attractive potential field
    attractive_potential = np.sqrt((goal_pose[0] - current_pose[0])**2 + (goal_pose[1] - current_pose[1])**2)

    # Compute the repulsive potential field based on lidar data
    repulsive_potential = 1.0 / lidar.get_sensor_values()

    # Combine the attractive and repulsive potential fields
    potential_field = attractive_potential + repulsive_potential

    # Compute the gradient of the potential field
    gradient = np.gradient(potential_field)
    
    # print("Gradient: ", gradient)

    # Compute the command based on the gradient
    command = {"forward": gradient[0],
               "rotation": gradient[1]}

    return command
