from __future__ import division
from os import link
import sim_update
import pybullet as p
import random
import numpy as np
import math
import matplotlib.pyplot as plt
import time
import threading

MAX_ITERS = 10000
delta_q = 0.1

def visualize_path(q_1, q_2, env, color=[0, 1, 0]):
    env.set_joint_positions(q_1)
    point_1 = list(p.getLinkState(env.robot_body_id, 6)[0])
    point_1[2] -= 0.15
    env.set_joint_positions(q_2)
    point_2 = list(p.getLinkState(env.robot_body_id, 6)[0])
    point_2[2] -= 0.15
    p.addUserDebugLine(point_1, point_2, color, 1.5)


def rrt(q_init, q_goal, MAX_ITERS, delta_q, steer_goal_p, env, distance=0.12):
    V, E = [q_init], []
    path, found = [], False
    for i in range(MAX_ITERS):
        q_rand = semi_random_sample(steer_goal_p, q_goal)
        q_nearest = nearest(V, q_rand)
        q_new = steer(q_nearest, q_rand, delta_q)
        if not env.check_collision(q_new, 0.12):
            if q_new not in V:
                V.append(q_new)
            if (q_nearest, q_new) not in E:
                E.append((q_nearest, q_new))
                visualize_path(q_nearest, q_new, env)
            if get_euclidean_distance(q_goal, q_new) < delta_q:
                V.append(q_goal)
                E.append((q_new, q_goal))
                visualize_path(q_new, q_goal, env)
                found = True
                break

    if found:
        current_q = q_goal
        path.append(current_q)
        while current_q != q_init:
            for edge in E:
                if edge[1] == current_q:
                    current_q = edge[0]
                    path.append(edge[0])
        path.reverse()
        return path
    else:
        return None


def semi_random_sample(steer_goal_p, q_goal):
    prob = random.random()

    if prob < steer_goal_p:
        return q_goal
    else:
        # Uniform sample over reachable joint angles
        q_rand = [random.uniform(-np.pi, np.pi) for i in range(len(q_goal))]
    return q_rand


def get_euclidean_distance(q1, q2):
    distance = 0
    for i in range(len(q1)):
        distance += (q2[i] - q1[i])**2
    return math.sqrt(distance)


def nearest(V, q_rand):
    distance = float("inf")
    q_nearest = None
    for idx, v in enumerate(V):
        if get_euclidean_distance(q_rand, v) < distance:
            q_nearest = v
            distance = get_euclidean_distance(q_rand, v)
    return q_nearest


def steer(q_nearest, q_rand, delta_q):
    q_new = None
    if get_euclidean_distance(q_rand, q_nearest) <= delta_q:
        q_new = q_rand
    else:
        q_hat = [(q_rand[i] - q_nearest[i]) / get_euclidean_distance(q_rand, q_nearest) for i in range(len(q_rand))]
        q_new = [q_nearest[i] + q_hat[i] * delta_q for i in range(len(q_hat))]
    return q_new


def get_grasp_position_angle(object_id):
    position, grasp_angle = np.zeros((3, 1)), 0
    position, orientation = p.getBasePositionAndOrientation(object_id)
    grasp_angle = p.getEulerFromQuaternion(orientation)[2]
    return position, grasp_angle

def run():
    num_trials = 20  
    path_lengths = []
    env.load_gripper()
    passed = 0
    for trial in range(num_trials):
        object_id = env._objects_body_ids[0]
        position, grasp_angle = get_grasp_position_angle(object_id)
        grasp_success = env.execute_grasp(position, grasp_angle)
        if grasp_success:
            path_conf = rrt(env.robot_home_joint_config,
                            env.robot_goal_joint_config, MAX_ITERS, delta_q, 0.5, env)
            if path_conf is None:
                print("No collision-free path is found within the time budget. Continuing...")
                path_lengths.append(None)  
            else:
                path_length = 0
                for i in range(1, len(path_conf)):
                    path_length += get_euclidean_distance(path_conf[i-1], path_conf[i])
                path_lengths.append(path_length)
                
                env.set_joint_positions(env.robot_home_joint_config)
                markers = []
                for joint_state in path_conf:
                    env.move_joints(joint_state, speed=0.05)
                    link_state = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)
                    # markers.append(sim_update.SphereMarker(link_state[0], radius=0.02))

                print("Path executed. Dropping the object")

                env.open_gripper()
                env.step_simulation(num_steps=5)
                env.close_gripper()

                for joint_state in reversed(path_conf):
                    env.move_joints(joint_state, speed=0.1)
                markers = None
            p.removeAllUserDebugItems()

        env.robot_go_home()

        object_pos, _ = p.getBasePositionAndOrientation(object_id)
        if object_pos[0] >= -0.8 and object_pos[0] <= -0.2 and\
            object_pos[1] >= -0.3 and object_pos[1] <= 0.3 and\
            object_pos[2] <= 0.2:
            passed += 1
        env.reset_objects()
def draw():
    print("Starting draw function")
    # Initialize line IDs
    line_ids = [None, None, None]
    current_object_id = None
    current_obstacle_id = None
    
    def get_distance(a, b):
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)
    
    while True:
        try:
            # Ensure there are objects and obstacles in the environment
            if len(env._objects_body_ids) == 0 or len(env.obstacles) == 0:
                print("No objects or obstacles found, waiting...")
                time.sleep(0.1)
                continue
            
            # Get current object and obstacle IDs
            object_id = env._objects_body_ids[0]
            obstacles_id = env.obstacles[0]
            
            # Check if the object or obstacle has been reset
            if object_id != current_object_id or obstacles_id != current_obstacle_id:
                # Reset line IDs when a new object or obstacle is detected
                line_ids = [None, None, None]
                current_object_id = object_id
                current_obstacle_id = obstacles_id
                print(f"Detected object or obstacles change. Updated object_id: {object_id}, obstacle_id: {obstacles_id}")
            
            # Verify that the object and obstacle still exist
            try:
                p.getBodyInfo(object_id)
                p.getBodyInfo(obstacles_id)
            except p.error as e:
                print(f"Body ID error: {e}")
                time.sleep(0.1)
                continue
            
            # Get link positions
            getlink1 = p.getLinkState(object_id, 0)[0]
            getlink2 = p.getLinkState(object_id, 1)[0]
            midpoint = np.add(getlink1, getlink2) / 2
            
            # Find the closest points between the object and obstacle
            closest_points = p.getClosestPoints(obstacles_id, object_id, 100)
            if not closest_points:
                print("No closest points found")
                a = getlink1  # Assign a default value to avoid errors
            else:
                a = closest_points[0][5]
                
            # Calculate the distance and print it
            distance = get_distance(midpoint, a)
            print(f"Distance between midpoint and closest point: {distance}")
            
            # Define lines to be drawn
            lines_to_draw = [
                (getlink1, a, [1, 0, 0]),    # Red line
                (getlink2, a, [1, 0, 0]),    # Green line
                (midpoint, a, [0, 1, 0])     # Green line
            ]
            
            # Draw or update debug lines
            for i, (start, end, color) in enumerate(lines_to_draw):
                if line_ids[i] is None:
                    # Create a new debug line if it doesn't exist
                    line_ids[i] = p.addUserDebugLine(start, end, lineColorRGB=color, lineWidth=2)
                else:
                    # Update the existing debug line
                    p.addUserDebugLine(start, end, lineColorRGB=color, lineWidth=2, replaceItemUniqueId=line_ids[i])
            
        except IndexError as e:
            print(f"IndexError: {e}. Possible object or obstacle indices out of range. Retrying...")
            # Reset line IDs to reinitialize lines in the next iteration
            line_ids = [None, None, None]
            current_object_id = None
            current_obstacle_id = None
        except Exception as e:
            print(f"Exception in draw: {e}")
        
        # Pause briefly to reduce CPU usage
        time.sleep(0.1)
if __name__ == "__main__":
    random.seed(5)
    object_shapes = [
        "assets/objects/rod.urdf",
    ]
    env = sim_update.PyBulletSim(object_shapes = object_shapes)
    thread1 = threading.Thread(target=run)
    thread2 = threading.Thread(target=draw)
    thread1.start()
    # thread2.start()
    