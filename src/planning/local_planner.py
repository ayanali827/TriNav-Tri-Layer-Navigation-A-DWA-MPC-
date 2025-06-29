import numpy as np
from utils.geometry import normalize_angle, distance

class DWAPlanner:
    def __init__(self, config):
        self.config = config
        
    def predict_trajectory(self, state, v, w, predict_time, dt, goal=None, goal_threshold=0.3):
        traj = [state]
        t = 0
        while t <= predict_time:
            x, y, yaw, v_curr, w_curr = traj[-1]
            
            if goal is not None:
                dist_to_goal = distance([x, y], goal)
                if dist_to_goal < goal_threshold:
                    break
            
            x += v_curr * np.cos(yaw) * dt
            y += v_curr * np.sin(yaw) * dt
            yaw += w_curr * dt
            yaw = normalize_angle(yaw)
            
            tau_v = 0.15
            tau_w = 0.15
            v_next = v_curr + (v - v_curr) * (dt / tau_v)
            w_next = w_curr + (w - w_curr) * (dt / tau_w)
            
            traj.append(np.array([x, y, yaw, v_next, w_next]))
            t += dt
        return np.array(traj)

    def goal_distance_cost(self, state, goal):
        return 1.0 / (1.0 + distance(state, goal))

    def goal_heading_cost(self, state, goal):
        dx = goal[0] - state[0]
        dy = goal[1] - state[1]
        goal_theta = np.arctan2(dy, dx)
        theta_diff = normalize_angle(goal_theta - state[2])
        return 1.0 - abs(theta_diff) / np.pi

    def calculate_obstacle_cost(self, traj, obstacles, robot_radius):
        min_distances = []
        for state in traj:
            pos = state[:2]
            distances = [distance(pos, obstacle[:2]) for obstacle in obstacles]
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

    def plan(self, current_state, goal, obstacles, global_path):
        cfg = self.config
        local_goal = goal
        path_angle_cost = 0
        
        if global_path is not None and len(global_path) > 1:
            current_pos = current_state[:2]
            distances = [distance(current_pos, p) for p in global_path]
            closest_idx = np.argmin(distances)
            
            lookahead_dist = max(0.5, min(2.0, current_state[3] * 1.0))
            lookahead_idx = closest_idx
            accumulated_dist = 0.0
            
            while lookahead_idx < len(global_path)-1 and accumulated_dist < lookahead_dist:
                segment_length = distance(global_path[lookahead_idx], global_path[lookahead_idx+1])
                accumulated_dist += segment_length
                lookahead_idx += 1
            
            local_goal = global_path[lookahead_idx]
            path_direction = np.arctan2(local_goal[1]-current_pos[1], local_goal[0]-current_pos[0])
            path_angle_cost = abs(normalize_angle(path_direction - current_state[2]))
        
        v_range = [
            max(cfg['min_speed'], current_state[3] - cfg['max_accel'] * cfg['dt']),
            min(cfg['max_speed'], current_state[3] + cfg['max_accel'] * cfg['dt'])
        ]
        w_range = [
            max(-cfg['max_yawrate'], current_state[4] - cfg['max_dyawrate'] * cfg['dt']),
            min(cfg['max_yawrate'], current_state[4] + cfg['max_dyawrate'] * cfg['dt'])
        ]
        
        v_samples = np.linspace(v_range[0], v_range[1], cfg['resolution'])
        w_samples = np.linspace(w_range[0], w_range[1], cfg['resolution'])
        
        best_score = float('-inf')
        best_v = 0.0
        best_w = 0.0
        best_traj = None
        
        for v in v_samples:
            for w in w_samples:
                if abs(w) > 0.2 and v < 0.05:
                    continue
                    
                if abs(v) > 0.01:
                    curvature = abs(w) / v
                    if curvature > 1.5:
                        continue
                
                traj = self.predict_trajectory(current_state, v, w, cfg['predict_time'], cfg['dt'], goal=local_goal)
                
                goal_cost = self.goal_distance_cost(traj[-1, :2], local_goal)
                speed_cost = v
                heading_cost = self.goal_heading_cost(traj[-1], local_goal)
                obstacle_cost = self.calculate_obstacle_cost(traj, obstacles, cfg['robot_radius'])
                
                if obstacle_cost == float('inf'):
                    continue
                
                total_cost = (cfg['weights']['goal'] * goal_cost + 
                             cfg['weights']['speed'] * speed_cost + 
                             cfg['weights']['heading'] * heading_cost - 
                             cfg['weights']['obstacle'] * obstacle_cost -
                             cfg['weights'].get('path', 1.5) * path_angle_cost)
                
                if total_cost > best_score:
                    best_score = total_cost
                    best_v = v
                    best_w = w
                    best_traj = traj
        
        if best_traj is None:
            obstacle_directions = []
            for obs in obstacles:
                obs_dir = np.arctan2(obs[1]-current_state[1], obs[0]-current_state[0])
                obstacle_directions.append(obs_dir)
            
            if obstacle_directions:
                best_w = np.sign(normalize_angle(np.mean(obstacle_directions) - current_state[2] + np.pi))
                best_w = np.clip(best_w * 1.0, -cfg['max_yawrate'], cfg['max_yawrate'])
                best_v = -0.1
            else:
                best_v = v_samples[len(v_samples)//2]
                best_w = w_samples[len(w_samples)//2]
                
            best_traj = self.predict_trajectory(current_state, best_v, best_w, cfg['predict_time'], cfg['dt'])
                    
        return best_v, best_w, best_traj