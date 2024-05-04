""" A set of robotics control functions """

import random
import numpy as np

def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    
    # TODO for TP1
    # Implemented a "follow the wall" algorithm, where the robot will try to follow the wall on its left, without collinding with it.
    
    lidar_values = lidar.get_sensor_values()
    lidar_angles = lidar.get_ray_angles()
    lidar_len = len(lidar_values)
    
    # INDEX:
    # Lidar values are right, middle, left
    # positive rotation is left, negative is right
    
    front_index = lidar_len // 2 # F (middle of the matrix)
    back_index = 0 # B
    
    right_index = lidar_len // 4 # R
    left_index = front_index + right_index # L
    
    right_back_index = lidar_len // 8 # RB
    left_back_index = lidar_len - right_back_index # LB
    
    # right_front_index = right_index + right_back_index # RF
    # left_front_index = left_index - left_back_index # LF
    
    #                 225   180    135
    #                  FL    F    FR
    #                        
    #                   \    |    /
    #                        ^
    #            270 L ---  [@]  ---  R 90
    #                      robot
    #                   /    |    \
    # 
    #                  BL    B     BR
    #                315     0     45
    
    is_front_wall = np.any(lidar_values[front_index - int(lidar_len*0.15):front_index + int(lidar_len*0.15)] < 30)
    
    is_left_wall = True if lidar_values[left_index] < 30 else False
    
    no_wall = np.all(lidar_values[:] > 50)
    
    
    ### WALL FOLLOWING ALGORITHM ###
    
    # If there is no wall, go random    
    if no_wall:
        # No wall, go random and find a wall
        rotation_speed = random.uniform(-1, 1)
        speed = 0.3
    
    # If there is a wall in front and no wall on the left, turn right going around the wall
    elif is_front_wall and not is_left_wall:
        rotation_speed = -0.3
        speed = -0.15
        
    # If there is a wall in front and on the left, turn right but without moving
    elif is_front_wall and is_left_wall:
        rotation_speed = -0.3
        speed = 0
    
    # If there is nothing in front but there is a wall left, follow the wall slightly turning left
    elif not is_front_wall and is_left_wall: 
        rotation_speed = 0.05
        speed = 0.3
        
    # If there is nothing in front and no wall on the left, follow the wall turning right
    elif not is_front_wall and not is_left_wall:
        rotation_speed = 0.325
        speed = 0.25
    
    # Else, should not happen in this logic
    else:
        rotation_speed = 1
        speed = 0
        
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
    
    # To calculate the forces applied in the robot path, we will get the vector to the closest wall and  te vector to the goal.
    
    #                 225   180    135
    #                  FL    F    FR
    #                        
    #                   \    |    /
    #                        ^
    #            270 L ---  [@]  ---  R 90
    #                      robot
    #                   /    |    \
    # 
    #                  BL    B     BR
    #                315     0     45
    
    lidar_values = lidar.get_sensor_values()
    lidar_angles = lidar.get_ray_angles()
    lidar_len = len(lidar_values)
    
    closest_wall = np.min(lidar_values)
    closest_wall_index = np.argmin(lidar_values)


    # Compute the command based on the gradient
    command = {"forward": gradient[0],
               "rotation": gradient[1]}

    return command
