""" A set of robotics control functions """

import random
import numpy as np

def polar_to_cartesian(r, theta):
    """
    Convert polar coordinates to cartesian
    r : float, radius
    theta : float, angle in radians
    """
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.array([x, y])

def cartesian_to_polar(x, y):
    """
    Convert cartesian coordinates to polar
    x : float, x coordinate
    y : float, y coordinate
    """
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    return r, theta

def d_goal_current_pose(p1, p2):
    """
    Calculate the distance between goal and current pose, knowing that they contain the angle in the third index
    p1 : np.array, point 1
    p2 : np.array, point 2
    """
    p1 = np.array([p1[0], p1[1]])
    p2 = np.array([p2[0], p2[1]])
    
    return np.linalg.norm(p1 - p2)

def gradient_calculator_closest_wall(lidar_values, lidar_angles, current_pose, goal_pose, K_obs = 1, K_goal = 1, d_safe = 300):
    """
    Calculate the gradient of the current robot location in the potential field
    lidar_values : np.array, lidar values
    lidar_angles : np.array, lidar angles
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    """
    
    """
    Attrative gradient:
            K_goal
    ∇f = ------------ (q_goal - q)
         d(q, q_goal)
    
    
    
    Repulsive gradient:
              K_obs
    ∇f = --------------- * ( (1 / d(q, q_obs)) - (1 / d_safe) ) * (q_obs - q))
           d^3(q, q_obs)
           
    """
    
    ## Calculate the repulsive gradient
    
    # Calculate the vector to the closest wall
    closest_wall_index = np.argmin(lidar_values)
    closest_wall_distance = lidar_values[closest_wall_index]
    closest_wall_angle = lidar_angles[closest_wall_index]
    wall_vector = polar_to_cartesian(closest_wall_distance, closest_wall_angle)
    
    # The repulsive gradient
    # The original formula was d^3, but i found it better to be d^2 and took in consideration the robot size as a horizontal shift to the function
    robot_size = 30
    if closest_wall_distance - robot_size <= 0:
        robot_size = closest_wall_distance - 1
    
    repulsive_gradient = K_obs / ((closest_wall_distance - robot_size)**2) * (1/(closest_wall_distance - robot_size) - 1/d_safe) * wall_vector
    
    ## Calculate the attractive gradient
    
    # Calculate the vector to the goal
    goal_vector = np.array([goal_pose[0] - current_pose[0], goal_pose[1] - current_pose[1]])
    
    attractive_gradient = K_goal / np.linalg.norm(goal_vector) * goal_vector

    # Return the total gradient
    return attractive_gradient + repulsive_gradient

def gradient_calculator_180field_of_lidar(lidar_values, lidar_angles, current_pose, goal_pose, K_obs = 1, K_goal = 1, d_safe = 300):
    """
    Calculate the gradient of the current robot location in the potential field using the front half of the lidar values (acctually a bit more than 180 degrees)
    """
    
    # Calculate the vector to the goal
    goal_vector = np.array([goal_pose[0] - current_pose[0], goal_pose[1] - current_pose[1]])
    attractive_gradient = K_goal / np.linalg.norm(goal_vector) * goal_vector

    # Initialize the total repulsive gradient
    total_repulsive_gradient = np.array([0.0, 0.0])
    
    # Constants
    robot_size = 20
    
    # Consider only the front half of the lidar values
    min_angle_index = np.argmin(np.abs(lidar_angles - (-np.pi/2*1.17))) # 1.17 is 210/180 and 210 is the human field of view (there is no real reason to use 210)
    max_angle_index = np.argmin(np.abs(lidar_angles - (np.pi/2*1.17)))
    front_values = lidar_values[min_angle_index:max_angle_index+1]
    front_angles = lidar_angles[min_angle_index:max_angle_index+1]
    
    # Calculate repulsive gradient for each lidar point
    for distance, angle in zip(front_values, front_angles):
        # Convert polar to cartesian coordinates for the obstacle vector
        wall_vector = polar_to_cartesian(distance, angle)

        # Adjust distance to consider robot size
        effective_distance = max(distance - robot_size, 1)  # avoid division by zero

        # Calculate repulsive gradient contribution from this obstacle
        if effective_distance < d_safe:
            # Modify the repulsive contribution based on the angle
            angle_weight = 1 / (abs(angle) + 1)**1.5  # make values closer to the literal front more relevant for repulsion
            repulsive_contribution = K_obs * angle_weight / (effective_distance**2) * (1/effective_distance - 1/d_safe) * wall_vector
            total_repulsive_gradient += repulsive_contribution

    # Return the total gradient (attractive + repulsive)
    return attractive_gradient + total_repulsive_gradient

def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    
    # TODO for TP1
    # Implemented a "follow the wall" algorithm, where the robot will try to follow the wall on its left, without collinding with it.
    # The way i handled the indexes where very bad, i reallized only when working in the potential field control.
    # I directly used the index instead of rellying on the angles values in lidar_anglesm, the logic is right bellow.
    
    # INDEX:
    # Lidar values are right, middle, left
    # positive rotation is left, negative is right
    
    #             len//2 + len//8    len//2     len//4 + len //8
    #                             FL    F    FR
    #                                   
    #                              \    |    /
    #                                   ^
    #     len//2 + len//4   270 L ---  [@]  ---  R len//4
    #                                 robot
    #                              /    |    \
    #            
    #                             BL    B     BR
    #                  len - len//8     0     len//8
    
    lidar_values = lidar.get_sensor_values()
    lidar_angles = lidar.get_ray_angles()
    lidar_len = len(lidar_values)
    
    front_index = lidar_len // 2 # F (middle of the matrix)
    back_index = 0 # B
    
    right_index = lidar_len // 4 # R
    left_index = front_index + right_index # L
    
    right_back_index = lidar_len // 8 # RB
    left_back_index = lidar_len - right_back_index # LB
    
    
    is_front_wall = np.any(lidar_values[front_index - int(lidar_len*0.15):front_index + int(lidar_len*0.15)] < 30)
    
    is_left_wall = True if lidar_values[left_index] < 30 else False
    
    no_wall = np.all(lidar_values[:] > 50)
    
    ### WALL FOLLOWING ALGORITHM ###
    
    # If there is no wall, go random and find a wall    
    if no_wall:
        rotation_speed = random.uniform(-1, 1)
        speed = 0.3
    
    # If there is a wall in front and no wall on the left, turn right
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
    
    # Calculate the gradient
    # gradient = gradient_calculator_closest_wall(lidar.get_sensor_values(), lidar.get_ray_angles(), current_pose, goal_pose, K_obs=400, K_goal=50, d_safe=300)
    gradient = gradient_calculator_180field_of_lidar(lidar.get_sensor_values(), lidar.get_ray_angles(), current_pose, goal_pose, K_obs=250, K_goal=50, d_safe=50)
    
    # Now we have the gradient, we can calculate the command to move the robot
    gradient_r, gradient_theta = cartesian_to_polar(gradient[0], gradient[1])
    
    # Force the result angle to be in the interval [-pi, pi]
    if((gradient_theta - current_pose[2]) > np.pi): gradient_theta -= 2*np.pi
    if((gradient_theta - current_pose[2]) < -np.pi): gradient_theta += 2*np.pi
    
    # Calculate the rotation speed
    rotation_speed = (gradient_theta - current_pose[2]) / np.pi
    
    # Limit the rotation speed to -1, 1
    if rotation_speed > 1:
        rotation_speed = 1
    elif rotation_speed < -1:
        rotation_speed = -1
    
    # Making the speed inversely proportional to the need to turn (if it needs to turn a lot, speed gets negative to go in reverse)
    dist_to_goal = d_goal_current_pose(current_pose, goal_pose)
    if dist_to_goal > 100:
        speed = (0.5 - abs(rotation_speed))*0.4
        
    # If the robot is close to the goal, slow down proportionally to the distance
    else: 
        speed = (0.5 - abs(rotation_speed))*(dist_to_goal/100)*0.4

    # If the robot is very close to the goal, stops
    if dist_to_goal < 10:
        speed = 0
        rotation_speed = 0
        
    command = {"forward": speed,
                "rotation": rotation_speed*0.5}
    
    return command
