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
delta_q = 0.1  # Step size

class Node:
    def __init__(self, joint_positions, parent=None):
        self.joint_positions = joint_positions
        self.parent = parent
def visualize_path(q_1, q_2, env, color=[0, 1, 0],tree='start'):
    env.set_joint_positions(q_1)
    point_1 = list(p.getLinkState(env.robot_body_id, 6)[0])
    point_1[2] -= 0.15
    
    env.set_joint_positions(q_2)
    point_2 = list(p.getLinkState(env.robot_body_id, 6)[0])
    point_2[2] -= 0.15
    debug_color = color if tree == 'start' else [1, 0, 1]  # Màu sắc khác nhau cho hai cây

    p.addUserDebugLine(point_1, point_2, debug_color, 1.5)


def bidirectional_rrt(env, q_start, q_goal, MAX_ITERS, delta_q, steer_goal_p, max_connection_distance=0.3):
    # Khởi tạo hai cây
    tree_start = [Node(q_start)]
    tree_goal = [Node(q_goal)]
    
    for i in range(MAX_ITERS):
        # Luân phiên mở rộng giữa cây start và cây goal
        if i % 2 == 0:
            current_tree = tree_start
            other_tree = tree_goal
            tree_identifier = 'start'
        else:
            current_tree = tree_goal
            other_tree = tree_start
            tree_identifier = 'goal'

        # Tăng xác suất kết nối với cây còn lại
        if random.random() < 0.3:  # 30% cơ hội nhắm thẳng vào cây còn lại
            nearest_other = nearest([node.joint_positions for node in other_tree], 
                                  current_tree[-1].joint_positions)
            q_rand = nearest_other
        else:
            q_rand = semi_random_sample(steer_goal_p, q_goal if tree_identifier == 'start' else q_start)

        # Tìm và mở rộng nút gần nhất
        q_nearest = nearest([node.joint_positions for node in current_tree], q_rand)
        q_new = steer(q_nearest, q_rand, delta_q)

        if not env.check_collision(q_new, distance=0.18):
            new_node = Node(q_new)
            nearest_node = next(node for node in current_tree if node.joint_positions == q_nearest)
            new_node.parent = nearest_node
            current_tree.append(new_node)
            visualize_path(q_nearest, q_new, env, tree=tree_identifier)

            # Thử kết nối với cây còn lại
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
    """Thử kết nối hai nút với bước nhỏ hơn"""
    distance = get_euclidean_distance(q1, q2)
    steps = max(int(distance / (step_size/2)), 5)  # Ít nhất 5 bước kiểm tra
    
    for i in range(1, steps):
        t = i / steps
        q_interp = [q1[j] + (q2[j] - q1[j]) * t for j in range(len(q1))]
        if env.check_collision(q_interp, distance=0.18):
            return False
        # Vẽ đường kết nối để debug
        visualize_path(q1, q_interp, env, color=[0, 0, 1])  # Màu xanh dương cho đường kết nối
    return True

def extract_path(tree1, tree2, node1, node2, env):
    """Trích xuất đường đi từ hai cây"""
    path_start = []
    current = node1
    while current is not None:
        path_start.append(current.joint_positions)
        current = current.parent
    path_start.reverse()

    path_goal = []
    current = node2
    while current is not None:
        path_goal.append(current.joint_positions)
        current = current.parent

    # Làm mịn đường đi
    complete_path = path_start + path_goal
    return smooth_path(complete_path, env)
def smooth_path(path, env, max_tries=50):
    """Làm mịn đường đi bằng cách loại bỏ các điểm thừa"""
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
    """
    Chạy thuật toán Bidirectional RRT trong mô phỏng và tính toán khoảng cách đường đi sau 50 lần chạy.
    """
    env.load_gripper()
    passed = 0
    num_trials = 50  # Số lần chạy thử nghiệm
    path_lengths = []  # Danh sách lưu trữ độ dài đường đi
    
    for trial in range(num_trials):
        print(f"Thử nghiệm thứ: {trial + 1}")
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
            print("Path configuration:", path_conf)
            if path_conf is None:
                print("No collision-free path is found within the iteration limit. Continuing ...")
                path_lengths.append(None)  # Ghi nhận không tìm thấy đường đi
            else:
                # Tính độ dài đường đi
                path_length = 0
                for i in range(1, len(path_conf)):
                    q_prev = path_conf[i-1]
                    q_curr = path_conf[i]
                    path_length += get_euclidean_distance(q_prev, q_curr)
                path_lengths.append(path_length)
                print(f"Độ dài đường đi: {path_length}")

                # Thực thi đường đi trong mô phỏng
                env.set_joint_positions(env.robot_home_joint_config)
                markers = []
                for joint_state in path_conf:
                    env.move_joints(joint_state, speed=0.01)
                    link_state = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)
                    # markers.append(sim.SphereMarker(link_state[0], radius=0.01))
                print("Path executed. Dropping the object")
                env.open_gripper()
                env.step_simulation(num_steps=5)
                env.close_gripper()

                # Truy ngược đường đi về vị trí ban đầu
                path_conf_reversed = path_conf[::-1]
                if path_conf_reversed:
                    for joint_state in path_conf_reversed:
                        env.move_joints(joint_state, speed=0.01)
                        link_state = p.getLinkState(env.robot_body_id, env.robot_end_effector_link_index)
                #         markers.append(sim.SphereMarker(link_state[0], radius=0.01))
                # markers = None
            p.removeAllUserDebugItems()

        env.robot_go_home()
        object_pos, _ = p.getBasePositionAndOrientation(object_id)
        if (-0.8 <= object_pos[0] <= -0.2) and (-0.3 <= object_pos[1] <= 0.3) and (object_pos[2] <= 0.2):
            passed += 1
        env.reset_objects()
    
    print(f"[Bidirectional RRT Object Transfer] {passed} / {num_trials} cases passed")
    
    # In kết quả độ dài đường đi
    successful_lengths = [length for length in path_lengths if length is not None]
    failed_trials = len([length for length in path_lengths if length is None])
    print(f"Số lần tìm thấy đường đi: {len(successful_lengths)}")
    print(f"Số lần không tìm thấy đường đi: {failed_trials}")
    if successful_lengths:
        average_length = sum(successful_lengths) / len(successful_lengths)
        print(f"Độ dài đường đi trung bình: {average_length:.4f}")
        print("lengths:",path_lengths)
    else:
        print("Không có đường đi nào được tìm thấy để tính trung bình.")

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
    # Tạo KD-Tree cho tree_goal
    tree_goal_positions = [node.joint_positions for node in tree_goal]
    kd_tree_goal = cKDTree(tree_goal_positions)

    for node_start in tree_start:
        q_start = node_start.joint_positions
        # Tìm tất cả các nút trong tree_goal gần node_start hơn max_connection_distance
        indices = kd_tree_goal.query_ball_point(q_start, r=max_connection_distance)
        for idx in indices:
            node_goal = tree_goal[idx]
            distance = get_euclidean_distance(q_start, node_goal.joint_positions)
            print(f"Đã tìm thấy cặp nút gần nhau với khoảng cách {distance:.4f}. Thử kết nối...")
            
            # Kiểm tra va chạm giữa hai nút
            steps = int(distance / (max_connection_distance / 2))
            collision = False
            for t in range(1, steps + 1):
                interp_q = [
                    q_start[j] + 
                    (node_goal.joint_positions[j] - q_start[j]) * (t / steps)
                    for j in range(len(q_start))
                ]
                if env.check_collision(interp_q, distance=0.15):
                    print("Đường đi giữa hai nút gặp va chạm. Không kết nối.")
                    collision = True
                    break
            if not collision:
                # Xây dựng đường đi từ cây start
                path_start = []
                current = node_start
                while current is not None:
                    path_start.append(current.joint_positions)
                    current = current.parent
                path_start.reverse()

                # Xây dựng đường đi từ cây goal
                path_goal = []
                current = node_goal
                while current is not None:
                    path_goal.append(current.joint_positions)
                    current = current.parent

                # Kết hợp cả hai đường đi
                combined_path = path_start + path_goal
                print("Hai cây đã được kết nối thành công.")
                return combined_path

    print("Không tìm thấy cặp nút nào để kết nối hai cây.")
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