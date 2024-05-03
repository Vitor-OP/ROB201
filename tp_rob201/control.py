""" A set of robotics control functions """

import random
import numpy as np

def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    
    # TODO for TP1
    
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
        print('1.0', end='\r')
        speed = -0.1
        rotation_speed = -0.4
    
    # Foward going (no obstacle in front)
    else: 
        # Slighltly turn right if going forward but close to a wall on the left
        if np.any(lidar_values[middle_index + int(lidar_len*0.05):middle_index + int(lidar_len*0.15)]) < 75: 
            print('2.0', end='\r')
            speed = 0.3
            rotation_speed = 0.2
        
        else: # no wall in left side
            # if np.all(lidar_values[middle_index + side_index - int(lidar_len*1):middle_index + side_index + int(lidar_len*0.2)]) > 200: 
                print('2.1', end='\r')
                speed = 0.4
                rotation_speed = -1
            # else:
            #     print('2.2', end='\r')
            #     speed = 0.1
            #     rotation_speed = -1
    
    # # Obstacle on right side, turn left
    # elif np.any(lidar_values[middle_index - int(lidar_len*0.1):middle_index - int(lidar_len*0.05)] < 50):
    #     print('3')
    #     speed = 0.1
    #     rotation_speed = 0.3
    
    # # Obstacle on left side, turn right
    # elif np.any(lidar_values[middle_index + int(lidar_len*0.05):middle_index + int(lidar_len*0.1)] < 50):
    #     print('4')
    #     speed = 0.1
    #     rotation_speed = 0.3
        
    # print(lidar_values[side_index], lidar_values[middle_index], lidar_values[lidar_len - side_index])

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
