from __future__ import division
import random
import math
import pybullet as p
import sim
import threading
import time
from scipy.spatial import cKDTree
import numpy as np

MAX_ITERS = 10000
delta_q = 0.2

class Node:
    def __init__(self, joint_positions, parent=None):
        self.joint_positions = joint_positions
        self.parent = parent

def visualize_path(q_1, q_2, env, color=[0, 1, 0], tree='start'):
    env.set_joint_positions(q_1)
    point_1 = p.getLinkState(env.robot_body_id, 6)[0]
    env.set_joint_positions(q_2)
    point_2 = p.getLinkState(env.robot_body_id, 6)[0]
    debug_color = color if tree == 'start' else [1, 0, 1]
    p.addUserDebugLine(point_1, point_2, debug_color, 1.0)

def attractive_force(q_current, q_goal, k_att=1.0):
    diff = np.array(q_goal) - np.array(q_current)
    return k_att * diff

def repulsive_force(q_current, env, k_rep=1000.0, d0=1):
    env.set_joint_positions(q_current)
    rep_force = np.zeros(6)
    end_effector_pos = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)[0]

    if hasattr(env, 'grasped_object_id') and env.grasped_object_id is not None:
        object_pos, object_orn = p.getBasePositionAndOrientation(env.grasped_object_id)
        check_points = get_object_check_points(object_pos, object_orn)

        for obstacle_id in env.obstacles:
            for point in check_points:
                closest_points = p.getClosestPoints(env.grasped_object_id, obstacle_id, d0)
                if closest_points:
                    d = closest_points[0][8]
                    if d < d0:
                        obstacle_pos = closest_points[0][6]
                        diff = np.array(point) - np.array(obstacle_pos)
                        direction = diff / np.linalg.norm(diff)
                        cart_force = k_rep * (1/d - 1/d0) * (1/d**3) * direction
                        object_to_ee_transform = np.array(end_effector_pos) - np.array(object_pos)
                        transformed_force = transform_force(cart_force, object_to_ee_transform)
                        J = calculate_jacobian(env, q_current)
                        joint_force = np.dot(J.T, transformed_force[:3])
                        rep_force += joint_force
                        p.addUserDebugLine(point, obstacle_pos, [1, 0, 0], lineWidth=2, lifeTime=0.1)
    return rep_force

def get_object_check_points(object_pos, object_orn):
    rot_matrix = p.getMatrixFromQuaternion(object_orn)
    rot_matrix = np.array(rot_matrix).reshape(3, 3)
    half_extents = np.array([0.1, 0.02, 0.02])
    points = []
    for x in [-1, 1]:
        for y in [-1, 1]:
            for z in [-1, 1]:
                point = np.array([
                    x * half_extents[0],
                    y * half_extents[1],
                    z * half_extents[2]
                ])
                rotated_point = np.dot(rot_matrix, point)
                world_point = rotated_point + np.array(object_pos)
                points.append(world_point)

    return points

def transform_force(force, transform):
    moment = np.cross(transform, force[:3])
    transformed_force = np.concatenate([force[:3], moment])
    return transformed_force

def calculate_jacobian(env, q):
    zero_vec = [0.0] * 6
    jac_t, _ = p.calculateJacobian(
        env.robot_body_id,
        env.robot_end_effector_link_index,
        [0, 0, 0],
        q,
        zero_vec,
        zero_vec
    )
    return np.array(jac_t)

def modified_steer_birrt(q_nearest, q_rand, delta_q, q_goal, env, tree_identifier):
    if tree_identifier == 'start':
        att = attractive_force(q_nearest, q_goal)
    else:
        att = attractive_force(q_nearest, env.robot_home_joint_config)
    rep = repulsive_force(q_nearest, env, k_rep=500.0, d0=1.0)
    total_force = att + rep
    if np.linalg.norm(total_force) > 0:
        total_force = total_force / np.linalg.norm(total_force)
    direction = np.array(q_rand) - np.array(q_nearest)
    if np.linalg.norm(direction) > 0:
        direction = direction / np.linalg.norm(direction)
    combined_direction = 0.6 * direction + 0.4 * total_force
    if np.linalg.norm(combined_direction) > 0:
        combined_direction = combined_direction / np.linalg.norm(combined_direction)
    q_new = np.array(q_nearest) + delta_q * combined_direction
    env.set_joint_positions(q_nearest)
    start_pos = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)[0]
    env.set_joint_positions(q_new)
    end_pos = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)[0]
    color = [0, 1, 0] if tree_identifier == 'start' else [1, 0, 1]
    p.addUserDebugLine(start_pos, end_pos, color, lineWidth=2, lifeTime=0.1)

    return q_new.tolist()

def bidirectional_rrt(env, q_start, q_goal, MAX_ITERS, delta_q, steer_goal_p, max_connection_distance=0.3):
    tree_start = [Node(q_start)]
    tree_goal = [Node(q_goal)]

    for i in range(MAX_ITERS):
        if i % 2 == 0:
            current_tree = tree_start
            other_tree = tree_goal
            tree_identifier = 'start'
        else:
            current_tree = tree_goal
            other_tree = tree_start
            tree_identifier = 'goal'

        if random.random() < 0.3:
            nearest_other = nearest([node.joint_positions for node in other_tree],
                                  current_tree[-1].joint_positions)
            q_rand = nearest_other
        else:
            q_rand = semi_random_sample(steer_goal_p, q_goal if tree_identifier == 'start' else q_start)
        q_nearest = nearest([node.joint_positions for node in current_tree], q_rand)
        q_new = modified_steer_birrt(q_nearest, q_rand, delta_q, q_goal, env, tree_identifier)
        if not env.check_collision(q_new, distance=0.165):
            new_node = Node(q_new)
            nearest_node = next(node for node in current_tree if node.joint_positions == q_nearest)
            new_node.parent = nearest_node
            current_tree.append(new_node)
            visualize_path(q_nearest, q_new, env, tree=tree_identifier)
            closest_other = nearest([node.joint_positions for node in other_tree], q_new)
            if get_euclidean_distance(q_new, closest_other) < max_connection_distance:
                if try_connect_nodes(q_new, closest_other, env, delta_q):
                    # Tìm thấy đường đi
                    if tree_identifier == 'start':
                        path = extract_path(current_tree, other_tree, new_node,
                                         next(node for node in other_tree
                                             if node.joint_positions == closest_other),
                                         env)
                    else:
                        path = extract_path(other_tree, current_tree,
                                         next(node for node in other_tree
                                             if node.joint_positions == closest_other),
                                         new_node, env)
                    return path

    return None

def try_connect_nodes(q1, q2, env, step_size):
    distance = get_euclidean_distance(q1, q2)
    steps = max(int(distance / (step_size/2)), 5)
    for i in range(1, steps):
        t = i / steps
        q_interp = [q1[j] + (q2[j] - q1[j]) * t for j in range(len(q1))]
        if env.check_collision(q_interp, distance=0.18):
            return False
        visualize_path(q1, q_interp, env, color=[0, 0, 1])
    return True

def extract_path(tree_start, tree_goal, node_start, node_goal,env):
    path_start = []
    current = node_start
    while current is not None:
        path_start.append(current.joint_positions)
        current = current.parent
    path_start.reverse()
    path_goal = []
    current = node_goal
    while current is not None:
        path_goal.append(current.joint_positions)
        current = current.parent
    complete_path = path_start + path_goal
    # return smooth_path(complete_path, env
    return complete_path
def smooth_path(path, env, max_tries=50):
    if len(path) <= 2:
        return path
    i = 0
    while i < len(path) - 2 and max_tries > 0:
        if try_connect_nodes(path[i], path[i + 2], env, delta_q/2):
            path.pop(i + 1)
            max_tries -= 1
        else:
            i += 1
    return path

def semi_random_sample(steer_goal_p, steer_target):
    prob = random.random()
    if prob < steer_goal_p:
        q_rand = steer_target
    else:
        q_rand = [random.uniform(-math.pi, math.pi) for _ in range(len(steer_target))]
    return q_rand

def get_euclidean_distance(q1, q2):
    distance = 0
    for i in range(len(q1)):
        distance += (q2[i] - q1[i])**2
    return math.sqrt(distance)

def nearest(V, q_rand):
    distance = float("inf")
    q_nearest = None
    for v in V:
        dist = get_euclidean_distance(q_rand, v)
        if dist < distance:
            q_nearest = v
            distance = dist
    return q_nearest

def steer(q_nearest, q_rand, delta_q):
    q_new = None
    distance = get_euclidean_distance(q_rand, q_nearest)
    if distance <= delta_q:
        q_new = q_rand
    else:
        q_hat = [(q_rand[i] - q_nearest[i]) / distance for i in range(len(q_rand))]
        q_new = [q_nearest[i] + q_hat[i] * delta_q for i in range(len(q_hat))]
    return q_new

def get_grasp_position_angle(object_id):
    position, grasp_angle = np.zeros((3, 1)), 0
    position, orientation = p.getBasePositionAndOrientation(object_id)
    grasp_angle = p.getEulerFromQuaternion(orientation)[2]
    return position, grasp_angle

def run_bidirectional_rrt():
    env.load_gripper()
    passed = 0
    num_trials = 50
    path_lengths = []

    for trial in range(num_trials):
        object_id = env._objects_body_ids[0]
        position, grasp_angle = get_grasp_position_angle(object_id)
        grasp_success = env.execute_grasp(position, grasp_angle)
        if grasp_success:
            path_conf = bidirectional_rrt(
                env,
                env.robot_home_joint_config,
                env.robot_goal_joint_config,
                MAX_ITERS,
                delta_q,
                steer_goal_p=0.5,
                max_connection_distance=0.15
            )
            if path_conf is None:
                print("No collision-free path is found within the iteration limit. Continuing ...")
                path_lengths.append(None)  # Ghi nhận không tìm thấy đường đi
            else:
                for joint_state in path_conf:
                    joint_degrees = [round(math.degrees(angle), 2) for angle in joint_state]
                for joint_state in path_conf:
                    env.set_joint_positions(joint_state)
                    end_pos = list(p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)[0])
                    end_pos[2] -= 0.15
                    end_pos_rounded = [round(coord, 3) for coord in end_pos]
                path_length = 0
                for i in range(1, len(path_conf)):
                    q_prev = path_conf[i-1]
                    q_curr = path_conf[i]
                    path_length += get_euclidean_distance(q_prev, q_curr)
                path_lengths.append(path_length)
                env.set_joint_positions(env.robot_home_joint_config)
                markers = []
                for joint_state in path_conf:
                    env.move_joints(joint_state, speed=0.01)
                    link_state = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)
                    markers.append(sim.SphereMarker(link_state[0], radius=0.01))
                env.open_gripper()
                env.step_simulation(num_steps=5)
                env.close_gripper()
                path_conf_reversed = path_conf[::-1]
                if path_conf_reversed:
                    for joint_state in path_conf_reversed:
                        env.move_joints(joint_state, speed=0.01)
                        link_state = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)
                        markers.append(sim.SphereMarker(link_state[0], radius=0.01))
                markers = None
            p.removeAllUserDebugItems()

        env.robot_go_home()
        object_pos, _ = p.getBasePositionAndOrientation(object_id)
        if (-0.8 <= object_pos[0] <= -0.2) and (-0.3 <= object_pos[1] <= 0.3) and (object_pos[2] <= 0.2):
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
            if len(env._objects_body_ids) == 0 or len(env.obstacles) == 0:
                print("No objects or obstacles found, waiting...")
                time.sleep(0.1)
                continue

            object_id = env._objects_body_ids[0]
            obstacles_id = env.obstacles[0]

            if object_id != current_object_id or obstacles_id != current_obstacle_id:
                line_ids = [None, None, None]
                current_object_id = object_id
                current_obstacle_id = obstacles_id
                print(f"Detected object or obstacles change. Updated object_id: {object_id}, obstacle_id: {obstacles_id}")

            try:
                p.getBodyInfo(object_id)
                p.getBodyInfo(obstacles_id)
            except p.error as e:
                print(f"Body ID error: {e}")
                time.sleep(0.1)
                continue

            getlink1 = p.getLinkState(object_id, 0)[0]
            getlink2 = p.getLinkState(object_id, 1)[0]
            midpoint = np.add(getlink1, getlink2) / 2

            closest_points = p.getClosestPoints(obstacles_id, object_id, 100)
            if not closest_points:
                print("No closest points found")
                a = getlink1
            else:
                a = closest_points[0][5]

            distance = get_distance(midpoint, a)
            lines_to_draw = [
                (getlink1, a, [1, 0, 0]),    # Red line
                (getlink2, a, [1, 0, 0]),    # Green line
                (midpoint, a, [0, 1, 0])     # Green line
            ]
            for i, (start, end, color) in enumerate(lines_to_draw):
                if line_ids[i] is None:
                    line_ids[i] = p.addUserDebugLine(start, end, lineColorRGB=color, lineWidth=2)
                else:
                    p.addUserDebugLine(start, end, lineColorRGB=color, lineWidth=2, replaceItemUniqueId=line_ids[i])

        except IndexError as e:
            print(f"IndexError: {e}. Possible object or obstacle indices out of range. Retrying...")
            line_ids = [None, None, None]
            current_object_id = None
            current_obstacle_id = None
        except Exception as e:
            print(f"Exception in draw: {e}")
        time.sleep(0.1)

def is_path_collision_free(env, q_start, q_end, step_size=0.01, distance=0.15):
    distance_between = get_euclidean_distance(q_start, q_end)
    steps = int(distance_between / step_size)
    for i in range(1, steps + 1):
        interpolated_q = [
            q_start[j] + (q_end[j] - q_start[j]) * (i / steps)
            for j in range(len(q_start))
        ]
        if env.check_collision(interpolated_q, distance=distance):
            return False
    return True

def connect_trees(tree_start, tree_goal, env, max_connection_distance):
    tree_goal_positions = [node.joint_positions for node in tree_goal]
    kd_tree_goal = cKDTree(tree_goal_positions)
    for node_start in tree_start:
        q_start = node_start.joint_positions
        indices = kd_tree_goal.query_ball_point(q_start, r=max_connection_distance)
        for idx in indices:
            node_goal = tree_goal[idx]
            distance = get_euclidean_distance(q_start, node_goal.joint_positions)
            steps = int(distance / (max_connection_distance / 2))
            collision = False
            for t in range(1, steps + 1):
                interp_q = [
                    q_start[j] + 
                    (node_goal.joint_positions[j] - q_start[j]) * (t / steps)
                    for j in range(len(q_start))
                ]
                if env.check_collision(interp_q, distance=0.18):
                    collision = True
                    break
            if not collision:
                path_start = []
                current = node_start
                while current is not None:
                    path_start.append(current.joint_positions)
                    current = current.parent
                path_start.reverse()

                path_goal = []
                current = node_goal
                while current is not None:
                    path_goal.append(current.joint_positions)
                    current = current.parent

                combined_path = path_start + path_goal
                return combined_path
    return None

if __name__ == "__main__":
    random.seed(1)
    object_shapes = [
        "assets/objects/rod.urdf",
    ]
    env = sim.PyBulletSim(object_shapes=object_shapes)
    thread1 = threading.Thread(target=run_bidirectional_rrt)
    thread2 = threading.Thread(target=draw)
    thread1.start()
    # thread2.start()