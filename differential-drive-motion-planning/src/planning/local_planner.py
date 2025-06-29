import numpy as np
from utils.geometry import normalize_angle

def predict_trajectory(state, v, w, predict_time, dt, goal=None, goal_threshold=0.3):
    traj = [state]
    t = 0
    while t <= predict_time:
        x, y, yaw, v_curr, w_curr = traj[-1]
        
        # Check if we've reached the local goal
        if goal is not None:
            dist_to_goal = np.linalg.norm([x - goal[0], y - goal[1]])
            if dist_to_goal < goal_threshold:
                break
        
        # Update position and orientation
        x += v_curr * np.cos(yaw) * dt
        y += v_curr * np.sin(yaw) * dt
        yaw += w_curr * dt
        yaw = normalize_angle(yaw)
        
        # Model velocity transition with time constant
        tau_v = 0.15
        tau_w = 0.15
        v_next = v_curr + (v - v_curr) * (dt / tau_v)
        w_next = w_curr + (w - w_curr) * (dt / tau_w)
        
        # Create new state
        new_state = np.array([x, y, yaw, v_next, w_next])
        traj.append(new_state)
        t += dt
    
    return np.array(traj)

def goal_distance_cost(state, goal):
    return 1.0 / (1.0 + np.linalg.norm(goal - state))

def goal_heading_cost(state, goal):
    dx = goal[0] - state[0]
    dy = goal[1] - state[1]
    goal_theta = np.arctan2(dy, dx)
    theta_diff = normalize_angle(goal_theta - state[2])
    return 1.0 - abs(theta_diff) / np.pi

def calculate_obstacle_cost(traj, obstacles, robot_radius):
    if obstacles.size == 0:
        return 0.0
        
    min_distances = []
    for state in traj:
        pos = state[:2]
        distances = [np.linalg.norm(pos - obstacle[:2]) for obstacle in obstacles]
        min_dist = min(distances) if distances else float('inf')
        min_distances.append(min_dist)
    
    overall_min_dist = min(min_distances) if min_distances else float('inf')
    safety_margin = robot_radius + 0.1
    
    if overall_min_dist < safety_margin:
        return float('inf')
    elif overall_min_dist < safety_margin + 0.5:
        return 1.0 / (overall_min_dist - safety_margin + 1e-6)
    else:
        return 0.0

def dwa_planner(current_state, goal, config, obstacles, global_path):
    max_speed = config['max_speed']
    min_speed = config['min_speed']
    max_yawrate = config['max_yawrate']
    max_accel = config['max_accel']
    max_dyawrate = config['max_dyawrate']
    dt = config['dt']
    predict_time = config['predict_time']
    resolution = config['resolution']
    weights = config['weights']
    robot_radius = config['robot_radius']
    
    # Calculate minimum distance to obstacles
    if obstacles.size > 0:
        dists = np.linalg.norm(current_state[:2] - obstacles[:, :2], axis=1)
        min_obstacle_dist = np.min(dists)
    else:
        min_obstacle_dist = float('inf')
    
    local_goal = goal
    path_angle_cost = 0
    path_following_cost = 0
    
    # Find the closest point on global path
    if global_path is not None and len(global_path) > 1:
        current_pos = current_state[:2]
        distances = [np.linalg.norm(np.array(p) - current_pos) for p in global_path]
        closest_idx = np.argmin(distances)
        
        # Lookahead point
        lookahead_dist = max(0.5, min(2.0, current_state[3] * 1.0))
        lookahead_idx = closest_idx
        accumulated_dist = 0.0
        
        while lookahead_idx < len(global_path)-1 and accumulated_dist < lookahead_dist:
            segment_length = np.linalg.norm(np.array(global_path[lookahead_idx+1]) - 
                            np.array(global_path[lookahead_idx]))
            accumulated_dist += segment_length
            lookahead_idx += 1
        
        local_goal = global_path[lookahead_idx]
        path_direction = np.arctan2(local_goal[1]-current_pos[1], local_goal[0]-current_pos[0])
        path_angle_cost = abs(normalize_angle(path_direction - current_state[2]))
        
        # Calculate path following cost (distance to path)
        if closest_idx < len(global_path) - 1:
            seg_start = np.array(global_path[closest_idx])
            seg_end = np.array(global_path[closest_idx+1])
            seg_vec = seg_end - seg_start
            seg_length = np.linalg.norm(seg_vec)
            
            if seg_length > 1e-6:
                seg_vec /= seg_length
                pt_vec = current_pos - seg_start
                proj = np.dot(pt_vec, seg_vec)
                proj = np.clip(proj, 0, seg_length)
                closest_pt = seg_start + proj * seg_vec
                path_following_cost = np.linalg.norm(current_pos - closest_pt)
            else:
                path_following_cost = np.linalg.norm(current_pos - seg_start)
        else:
            path_following_cost = np.linalg.norm(current_pos - global_path[-1])
    
    v_range = [
        max(min_speed, current_state[3] - max_accel * dt),
        min(max_speed, current_state[3] + max_accel * dt)
    ]
    w_range = [
        max(-max_yawrate, current_state[4] - max_dyawrate * dt),
        min(max_yawrate, current_state[4] + max_dyawrate * dt)
    ]
    
    v_samples = np.linspace(v_range[0], v_range[1], resolution)
    w_samples = np.linspace(w_range[0], w_range[1], resolution)
    
    best_score = float('-inf')
    best_v = 0.0
    best_w = 0.0
    best_traj = None
    
    # Calculate adaptive speed penalty factors
    speed_penalty_factor = max(0.1, min(1.0, (min_obstacle_dist - robot_radius) / 2.0))
    goal_dist = np.linalg.norm(current_state[:2] - goal)
    goal_factor = max(0.2, min(1.0, goal_dist / 3.0))
    
    for v in v_samples:
        # Skip dangerous high speeds near obstacles
        if min_obstacle_dist < robot_radius + 0.5 and v > 0.5 * config['max_speed']:
            continue
            
        for w in w_samples:
            if abs(w) > 0.2 and v < 0.05:
                continue
                
            if abs(v) > 0.01:
                curvature = abs(w) / v
                if curvature > 1.5:
                    continue
            
            traj = predict_trajectory(current_state, v, w, predict_time, dt, goal=local_goal)
            
            goal_cost = goal_distance_cost(traj[-1, :2], local_goal)
            heading_cost = goal_heading_cost(traj[-1], local_goal)
            obstacle_cost = calculate_obstacle_cost(traj, obstacles, robot_radius)
            
            if obstacle_cost == float('inf'):
                continue
            
            # Modified cost function with adaptive speed penalty
            total_cost = (weights['goal'] * goal_cost +
                         weights['speed'] * v * speed_penalty_factor * goal_factor +
                         weights['heading'] * heading_cost - 
                         weights['obstacle'] * obstacle_cost -
                         weights.get('path', 1.5) * path_angle_cost -
                         weights.get('path_follow', 1.0) * path_following_cost)
            
            if total_cost > best_score:
                best_score = total_cost
                best_v = v
                best_w = w
                best_traj = traj
    
    if best_traj is None:
        # Try to move away from obstacles
        if min_obstacle_dist < float('inf'):
            # Calculate direction away from closest obstacle
            closest_obs = obstacles[np.argmin(dists)]
            obs_dir = np.arctan2(closest_obs[1]-current_state[1], closest_obs[0]-current_state[0])
            escape_dir = normalize_angle(obs_dir + np.pi)
            escape_angle = normalize_angle(escape_dir - current_state[2])
            
            # Set angular velocity to move away
            best_w = np.clip(escape_angle, -config['max_yawrate'], config['max_yawrate'])
            best_v = -0.2 if min_obstacle_dist < robot_radius + 0.3 else 0.2
        else:
            # Just rotate in place to find a way out
            best_w = config['max_yawrate'] * 0.5
            best_v = 0.0
            
        best_traj = predict_trajectory(current_state, best_v, best_w, predict_time, dt)
                
    return best_v, best_w, best_traj