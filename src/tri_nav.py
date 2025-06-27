import mujoco
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import time
import math
import heapq
from collections import deque
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

class AStarPlanner:
    def __init__(self, grid_size=0.2, robot_radius=0.4):
        self.grid_size = grid_size
        self.robot_radius = robot_radius
        
    def create_grid(self, obstacles, xlim=(-10, 10), ylim=(-5, 15)):
        min_x, max_x = xlim
        min_y, max_y = ylim
        width = int((max_x - min_x) / self.grid_size) + 1
        height = int((max_y - min_y) / self.grid_size) + 1
        
        grid = np.zeros((width, height))
        
        for obs in obstacles:
            x, y, r = obs
            obs_min_x = int((x - r - min_x) / self.grid_size)
            obs_max_x = int((x + r - min_x) / self.grid_size + 1)
            obs_min_y = int((y - r - min_y) / self.grid_size)
            obs_max_y = int((y + r - min_y) / self.grid_size + 1)
            
            obs_min_x = max(0, min(width-1, obs_min_x))
            obs_max_x = max(0, min(width-1, obs_max_x))
            obs_min_y = max(0, min(height-1, obs_min_y))
            obs_max_y = max(0, min(height-1, obs_max_y))
            
            grid[obs_min_x:obs_max_x+1, obs_min_y:obs_max_y+1] = 1
        
        return grid, (min_x, min_y)
    
    def world_to_grid(self, pos, grid_origin):
        x, y = pos
        min_x, min_y = grid_origin
        grid_x = int((x - min_x) / self.grid_size)
        grid_y = int((y - min_y) / self.grid_size)
        return (grid_x, grid_y)
    
    def grid_to_world(self, grid_pos, grid_origin):
        grid_x, grid_y = grid_pos
        min_x, min_y = grid_origin
        x = min_x + grid_x * self.grid_size
        y = min_y + grid_y * self.grid_size
        return (x, y)
    
    def heuristic(self, a, b):
        return np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def a_star_search(self, start, goal, grid, grid_origin):
        start = self.world_to_grid(start, grid_origin)
        goal = self.world_to_grid(goal, grid_origin)
        
        if grid[start[0], start[1]] == 1 or grid[goal[0], goal[1]] == 1:
            return None
        
        neighbors = [(0,1),(1,0),(0,-1),(-1,0),(1,1),(1,-1),(-1,1),(-1,-1)]
        
        close_set = set()
        came_from = {}
        gscore = {start:0}
        fscore = {start:self.heuristic(start, goal)}
        oheap = []
        heapq.heappush(oheap, (fscore[start], start))
        
        while oheap:
            current = heapq.heappop(oheap)[1]
            
            if current == goal:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path = path[::-1]
                return [self.grid_to_world(p, grid_origin) for p in path]
            
            close_set.add(current)
            for i,j in neighbors:
                neighbor = current[0]+i, current[1]+j
                
                if 0 <= neighbor[0] < grid.shape[0] and 0 <= neighbor[1] < grid.shape[1]:
                    if grid[neighbor[0], neighbor[1]] == 1:
                        continue
                    
                    tentative_g_score = gscore[current] + self.heuristic(current, neighbor)
                    
                    if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, float('inf')):
                        continue
                    
                    if tentative_g_score < gscore.get(neighbor, float('inf')) or neighbor not in [i[1] for i in oheap]:
                        came_from[neighbor] = current
                        gscore[neighbor] = tentative_g_score
                        fscore[neighbor] = tentative_g_score + self.heuristic(neighbor, goal)
                        heapq.heappush(oheap, (fscore[neighbor], neighbor))
        
        return None
    
    def simplify_path(self, path):
        if len(path) < 3:
            return path
            
        def perpendicular_distance(point, line_start, line_end):
            if np.allclose(line_start, line_end):
                return np.linalg.norm(point - line_start)
            return np.abs(np.cross(line_end-line_start, line_start-point)) / np.linalg.norm(line_end-line_start)
            
        max_dist = 0
        index = 0
        end = len(path) - 1
        
        for i in range(1, end):
            dist = perpendicular_distance(np.array(path[i]), 
                                         np.array(path[0]), 
                                         np.array(path[end]))
            if dist > max_dist:
                index = i
                max_dist = dist
                
        if max_dist > self.grid_size * 0.9:  # Less aggressive simplification
            rec_results1 = self.simplify_path(path[:index+1])
            rec_results2 = self.simplify_path(path[index:])
            return rec_results1[:-1] + rec_results2
        else:
            return [path[0], path[-1]]
    
    def plan_path(self, start, goal, obstacles):
        # Dynamically calculate grid bounds
        xs = [start[0], goal[0]]
        ys = [start[1], goal[1]]
        
        if obstacles is not None and len(obstacles) > 0:
            for obs in obstacles:
                xs.append(obs[0] - obs[2])  # Account for obstacle radius
                xs.append(obs[0] + obs[2])
                ys.append(obs[1] - obs[2])
                ys.append(obs[1] + obs[2])
        
        padding = self.robot_radius + 0.5
        xlim = (min(xs) - padding, max(xs) + padding)
        ylim = (min(ys) - padding, max(ys) + padding)
        
        grid, grid_origin = self.create_grid(obstacles, xlim=xlim, ylim=ylim)
        path = self.a_star_search(start, goal, grid, grid_origin)
        if path:
            return self.simplify_path(path)
        return None

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def get_car_state(data):
    x = data.qpos[0]
    y = data.qpos[1]
    quat = data.qpos[3:7]
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    yaw = rot.as_euler('zyx')[0]
    vx_global = data.qvel[0]
    vy_global = data.qvel[1]
    vx_body = vx_global * np.cos(yaw) + vy_global * np.sin(yaw)
    wz = data.qvel[5]
    return np.array([x, y, yaw, vx_body, wz])

def extract_obstacles(model):
    obstacles = []
    for i in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
        if geom_name and "obstacle" in geom_name:
            pos = model.geom_pos[i]
            size = model.geom_size[i]
            circumradius = math.sqrt(size[0]**2 + size[1]**2)
            obstacles.append([pos[0], pos[1], circumradius])
    return np.array(obstacles)

def check_collision(model, data):
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
        geom2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
        
        if geom1_name is None or geom2_name is None:
            continue
            
        if ("chasis" in geom1_name and "obstacle" in geom2_name) or \
           ("chasis" in geom2_name and "obstacle" in geom1_name):
            return True
    return False

def calculate_obstacle_cost(traj, obstacles, robot_radius):
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
    
    local_goal = goal
    path_angle_cost = 0
    if global_path is not None and len(global_path) > 1:
        current_pos = current_state[:2]
        distances = [np.linalg.norm(np.array(p) - current_pos) for p in global_path]
        closest_idx = np.argmin(distances)
        
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
    
    for v in v_samples:
        for w in w_samples:
            if abs(w) > 0.2 and v < 0.05:
                continue
                
            if abs(v) > 0.01:
                curvature = abs(w) / v
                if curvature > 1.5:
                    continue
            
            traj = predict_trajectory(current_state, v, w, predict_time, dt, goal=local_goal)
            
            goal_cost = goal_distance_cost(traj[-1, :2], local_goal)
            speed_cost = v
            heading_cost = goal_heading_cost(traj[-1], local_goal)
            obstacle_cost = calculate_obstacle_cost(traj, obstacles, robot_radius)
            
            if obstacle_cost == float('inf'):
                continue
            
            total_cost = (weights['goal'] * goal_cost + 
                         weights['speed'] * speed_cost + 
                         weights['heading'] * heading_cost - 
                         weights['obstacle'] * obstacle_cost -
                         weights.get('path', 1.5) * path_angle_cost)
            
            if total_cost > best_score:
                best_score = total_cost
                best_v = v
                best_w = w
                best_traj = traj
    
    if best_traj is None:
        print("DWA: No valid trajectory - attempting recovery behavior")
        obstacle_directions = []
        for obs in obstacles:
            obs_dir = np.arctan2(obs[1]-current_state[1], obs[0]-current_state[0])
            obstacle_directions.append(obs_dir)
        
        if obstacle_directions:
            best_w = np.sign(normalize_angle(np.mean(obstacle_directions) - current_state[2] + np.pi))
            best_w = np.clip(best_w * 1.0, -config['max_yawrate'], config['max_yawrate'])
            best_v = -0.1
        else:
            best_w = w
            best_v = v
            
        best_traj = predict_trajectory(current_state, best_v, best_w, predict_time, dt)
                
    return best_v, best_w, best_traj

def predict_trajectory(state, v, w, predict_time, dt, goal=None, goal_threshold=0.3):
    traj = [state]
    t = 0
    while t <= predict_time:
        x, y, yaw, v_curr, w_curr = traj[-1]
        
        if goal is not None:
            dist_to_goal = np.linalg.norm([x - goal[0], y - goal[1]])
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

def goal_distance_cost(state, goal):
    return 1.0 / (1.0 + np.linalg.norm(goal - state))

def goal_heading_cost(state, goal):
    dx = goal[0] - state[0]
    dy = goal[1] - state[1]
    goal_theta = np.arctan2(dy, dx)
    theta_diff = normalize_angle(goal_theta - state[2])
    return 1.0 - abs(theta_diff) / np.pi  # More aggressive penalty

def car_kinematic_model(state, v_cmd, w_cmd, dt):
    x, y, yaw, v_curr, w_curr = state
    tau_v = 0.2
    tau_w = 0.1
    
    v_next = v_curr + (v_cmd - v_curr) * (dt / tau_v)
    w_next = w_curr + (w_cmd - w_curr) * (dt / tau_w)
    
    x_new = x + v_next * np.cos(yaw) * dt
    y_new = y + v_next * np.sin(yaw) * dt
    yaw_new = yaw + w_next * dt
    
    return np.array([x_new, y_new, normalize_angle(yaw_new), v_next, w_next])

def mpc_controller(current_state, ref_trajectory, v_dwa, w_dwa, config, obstacles):
    horizon = config['horizon']
    dt_mpc = config['dt_mpc']
    max_speed = config['max_speed']
    max_yawrate = config['max_yawrate']
    weights = config['weights_mpc']
    robot_radius = config['robot_radius']
    
    if len(ref_trajectory) < horizon:
        last_state = ref_trajectory[-1] if len(ref_trajectory) > 0 else current_state
        padding = np.tile(last_state, (horizon - len(ref_trajectory), 1))
        ref_traj = np.vstack([ref_trajectory, padding])
    else:
        ref_traj = ref_trajectory[:horizon]
    
    def cost_function(u):
        v_cmds = u[:horizon]
        w_cmds = u[horizon:]
        cost = 0.0
        state = current_state.copy()
        
        for t in range(horizon):
            state = car_kinematic_model(state, v_cmds[t], w_cmds[t], dt_mpc)
            ref_state = ref_traj[t]
            
            pos_error = np.linalg.norm(state[:2] - ref_state[:2])
            cost += weights['position'] * pos_error
            
            heading_error = abs(normalize_angle(state[2] - ref_state[2]))
            cost += weights['heading'] * heading_error
            
            obstacle_cost = calculate_obstacle_cost([state], obstacles, robot_radius)
            cost += weights['obstacle'] * obstacle_cost
            
            if t > 0:
                dv = v_cmds[t] - v_cmds[t-1]
                dw = w_cmds[t] - w_cmds[t-1]
                cost += weights['dv'] * dv**2 + weights['dw'] * dw**2
            
            cost += weights['v'] * (v_cmds[t] - v_dwa)**2
            cost += weights['w'] * (w_cmds[t] - w_dwa)**2
        
        return cost
    
    v_bounds = [(config['min_speed'], config['max_speed'])] * horizon
    w_bounds = [(-config['max_yawrate'], config['max_yawrate'])] * horizon
    bounds = v_bounds + w_bounds
    
    u0 = np.concatenate([np.ones(horizon) * v_dwa, 
                         np.ones(horizon) * w_dwa])
    
    res = minimize(cost_function, u0, method='SLSQP', bounds=bounds, 
                   options={'maxiter': 50, 'ftol': 1e-4})
    
    if not res.success:
        print(f"MPC optimization failed: {res.message}")
        return v_dwa, w_dwa
    
    v_opt = res.x[0]
    w_opt = res.x[horizon]
    return v_opt, w_opt

# ========== MAIN SIMULATION ========== #
start_pos = [0, 0]
start_yaw = np.pi/2
goal_pos = [15, 12]
max_steps = 1000
goal_threshold = 0.3
high_level_freq = 2

obstacles = np.array([
    [1.2, 10.8, 0.6],    # Top-left
    [16.8, 1.2, 0.7],    # Bottom-right
    [2.5, 2.5, 0.5],     # Bottom-left
    [15.5, 10.5, 0.6],   # Top-right
    [1.0, 6.0, 0.7],     # Left-center
    [17.0, 6.0, 0.6],    # Right-center
    [4.0, 11.0, 0.5],    # Top-edge
    [14.0, 1.0, 0.7],    # Bottom-edge
    [3.0, 8.5, 0.6],     # Top-left quadrant
    [15.0, 3.5, 0.5],    # Bottom-right quadrant
    [8.0, 1.5, 0.7],     # Bottom-center
    [10.0, 10.5, 0.6],   # Top-center
    [1.5, 3.5, 0.5],     # Bottom-left area
    [16.5, 8.5, 0.7],    # Top-right area
    [5.5, 5.0, 0.6],     # Center-left
    [12.5, 7.0, 0.5],    # Center-right
    [9.0, 8.0, 0.7],     # Upper-center
    [9.0, 4.0, 0.6],     # Lower-center
    [3.5, 6.5, 0.5],     # Left area
    [14.5, 5.5, 0.7],    # Right area
    [7.0, 9.0, 0.6],     # Upper-left
    [11.0, 3.0, 0.5],    # Lower-right
    [6.0, 2.0, 0.7],     # Bottom-middle
    [12.0, 9.0, 0.6],    # Top-middle
    [8.5, 6.0, 0.5]      # Center
])


astar_planner = AStarPlanner(grid_size=0.3, robot_radius=0.4)
global_path = astar_planner.plan_path(start_pos, goal_pos, obstacles)

dwa_config = {
    'max_speed': 4,
    'min_speed': -0.5,
    'max_yawrate': 10,
    'max_accel': 0.5,
    'max_dyawrate': 7,
    'dt': 0.08,
    'predict_time': 2,
    'resolution': 12,
    'weights': {'goal': 2.5, 'speed': 0.2, 'heading': 0.1, 'obstacle': 0.6, 'path': 1.5},
    'robot_radius': 0.3
}

mpc_config = {
    'horizon': 2,
    'dt_mpc': 0.05,
    'max_speed': 4,
    'min_speed': -0.5,
    'max_yawrate': 7,
    'weights_mpc': {
        'position': 1.5,
        'heading': 0.1,
        'v': 0.4,
        'w': 0.4,
        'dv': 1.5,
        'dw': 0.7,
        'obstacle': 0.6
    }
}

config = {**dwa_config, **mpc_config}

model = mujoco.MjModel.from_xml_path("C:\\Users\\ayans\\Desktop\\Project\\Diiferential Drive Robot Model (DDR).xml")
data = mujoco.MjData(model)
model.opt.timestep = 0.02

data.qpos[0:2] = start_pos
quat = R.from_euler('z', start_yaw).as_quat()
data.qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]
data.qvel[:] = 0

trajectory = []
controls = []
timestamps = []
dwa_commands = []
mpc_commands = []
desired_v = 0.0
desired_w = 0.0
start_time = time.time()

prev_v_error = 0
integral_v = 0
prev_w_error = 0
integral_w = 0

prev_desired_v = 0
prev_desired_w = 0
filter_alpha = 0.8

try:
    from mujoco import viewer
    viewer_available = True
    viewer = viewer.launch_passive(model, data)
    viewer.sync()
except Exception as e:
    print(f"Could not create viewer: {e}")
    viewer_available = False

plt.ion()
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title('Live Trajectory Visualization')
ax.set_xlabel('X Position (m)')
ax.set_ylabel('Y Position (m)')
ax.grid(True)
ax.axis('equal')
ax.set_xlim(-10, 10)
ax.set_ylim(-5, 15)
start_marker, = ax.plot([start_pos[0]], [start_pos[1]], 'go', markersize=10, label='Start')
goal_marker, = ax.plot([goal_pos[0]], [goal_pos[1]], 'ro', markersize=10, label='Goal')

for obs in obstacles:
    circle = plt.Circle((obs[0], obs[1]), obs[2], color='r', alpha=0.5)
    ax.add_patch(circle)

if global_path is not None:
    global_path_line, = ax.plot([p[0] for p in global_path], [p[1] for p in global_path], 
                               'g--', alpha=0.7, label='Global Path')

traj_line, = ax.plot([], [], 'b-', label='Path')
dwa_traj_line, = ax.plot([], [], 'c--', alpha=0.5, label='DWA Plan')
mpc_traj_line, = ax.plot([], [], 'm--', alpha=0.7, label='MPC Plan')
ax.legend()
fig.canvas.draw()

for step in range(max_steps):
    state = get_car_state(data)
    trajectory.append(state)
    
    dist_to_goal = np.linalg.norm(state[:2] - goal_pos)
    if dist_to_goal < goal_threshold:
        print(f"Goal reached in {step} steps!")
        timestamps.append(step * model.opt.timestep)
        break
    
    timestamps.append(step * model.opt.timestep)

    slowdown_factor = min(1.0, dist_to_goal / 0.3)
    
    if step % high_level_freq == 0:
        local_config = config.copy()
        local_config['max_speed'] = config['max_speed'] * slowdown_factor
        local_config['max_yawrate'] = config['max_yawrate'] * slowdown_factor
        
        v_dwa, w_dwa, best_traj = dwa_planner(state, goal_pos, local_config, obstacles, global_path)
        dwa_commands.append((v_dwa, w_dwa))
        
        if best_traj is not None:
            ref_traj = best_traj[1:1+config['horizon']]
            if len(ref_traj) < config['horizon']:
                padding = np.tile(ref_traj[-1], (config['horizon'] - len(ref_traj), 1))
                ref_traj = np.vstack([ref_traj, padding])
        else:
            ref_traj = np.tile(state, (config['horizon'], 1))
        
        desired_v, desired_w = mpc_controller(
            state, ref_traj, v_dwa, w_dwa, config, obstacles
        )
        mpc_commands.append((desired_v, desired_w))
        
        desired_v = filter_alpha * prev_desired_v + (1-filter_alpha) * desired_v
        desired_w = filter_alpha * prev_desired_w + (1-filter_alpha) * desired_w
        prev_desired_v = desired_v
        prev_desired_w = desired_w
    
    dt_sim = model.opt.timestep
    v_error = desired_v - state[3]
    w_error = desired_w - state[4]
    
    integral_v += v_error * dt_sim
    derivative_v = (v_error - prev_v_error) / dt_sim
    u_forward = 1.2*v_error + 0.2*integral_v + 0.05*derivative_v
    
    integral_w += w_error * dt_sim
    derivative_w = (w_error - prev_w_error) / dt_sim
    u_turn = 0.8*w_error + 0.5*integral_w + 0.02*derivative_w
    
    prev_v_error = v_error
    prev_w_error = w_error
    
    u_forward = np.clip(u_forward, -1, 1)
    u_turn = np.clip(u_turn, -1, 1)
    
    data.ctrl[0] = u_forward
    data.ctrl[1] = u_turn
    
    controls.append([u_forward, u_turn])
    
    mujoco.mj_step(model, data)
    
    if viewer_available:
        viewer.sync()
        time.sleep(model.opt.timestep * 0.5)
    
    if step % 10 == 0 and len(trajectory) > 1:
        while len(ax.patches) > len(obstacles):
            ax.patches[-1].remove()
        
        traj_x = [s[0] for s in trajectory]
        traj_y = [s[1] for s in trajectory]
        traj_line.set_data(traj_x, traj_y)
        
        if 'best_traj' in locals():
            dwa_traj_line.set_data(best_traj[:,0], best_traj[:,1])
        
        if step % 50 == 0:
            current_v_range = [max(config['min_speed'], state[3] - config['max_accel'] * config['dt']),
                             min(config['max_speed'], state[3] + config['max_accel'] * config['dt'])]
            current_w_range = [max(-config['max_yawrate'], state[4] - config['max_dyawrate'] * config['dt']),
                             min(config['max_yawrate'], state[4] + config['max_dyawrate'] * config['dt'])]
    
            for v in np.linspace(current_v_range[0], current_v_range[1], 3):
                for w in np.linspace(current_w_range[0], current_w_range[1], 3):
                    traj = predict_trajectory(state, v, w, config['predict_time']/2, config['dt'])
                    ax.plot(traj[:,0], traj[:,1], 'y-', alpha=0.1)
        
        fig.canvas.draw()
        fig.canvas.flush_events()

if viewer_available:
    viewer.close()

trajectory = np.array(trajectory)
controls = np.array(controls)
dwa_commands = np.array(dwa_commands)
mpc_commands = np.array(mpc_commands)

plt.ioff()
plt.figure(figsize=(18, 12))

plt.subplot(2, 3, 1)
plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Actual Path')
plt.plot(start_pos[0], start_pos[1], 'go', markersize=10, label='Start')
plt.plot(goal_pos[0], goal_pos[1], 'ro', markersize=10, label='Goal')

for obs in obstacles:
    circle = plt.Circle((obs[0], obs[1]), obs[2], color='r', alpha=0.5)
    plt.gca().add_patch(circle)

if global_path is not None:
    plt.plot([p[0] for p in global_path], [p[1] for p in global_path], 
             'g--', alpha=0.7, label='Global Path')

plt.title('Robot Trajectory with Obstacles')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.axis('equal')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 2)
plt.plot(timestamps, np.degrees(trajectory[:, 2]), 'g-')
plt.title('Robot Heading')
plt.xlabel('Time (s)')
plt.ylabel('Heading (degrees)')
plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(timestamps, trajectory[:, 3], 'b-', label='Linear Velocity')
plt.plot(timestamps, trajectory[:, 4], 'r-', label='Angular Velocity')
plt.title('Robot Velocities')
plt.xlabel('Time (s)')
plt.ylabel('Velocity')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 4)
min_length = min(len(timestamps), len(controls))
plt.plot(timestamps[:min_length], controls[:min_length, 0], 'b-', label='Forward Control')
plt.plot(timestamps[:min_length], controls[:min_length, 1], 'r-', label='Turn Control')
plt.title('Control Signals')
plt.xlabel('Time (s)')
plt.ylabel('Control Value')
plt.ylim(-1.1, 1.1)
plt.grid(True)
plt.legend()

hl_times = np.array([i * high_level_freq * model.opt.timestep 
                    for i in range(len(dwa_commands))])

plt.subplot(2, 3, 5)
plt.plot(hl_times, dwa_commands[:, 0], 'bo-', label='DWA Velocity')
plt.plot(hl_times, mpc_commands[:, 0], 'ro-', label='MPC Velocity')
plt.title('DWA vs MPC Velocity Commands')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.grid(True)
plt.legend()

plt.subplot(2, 3, 6)
plt.plot(hl_times, dwa_commands[:, 1], 'bo-', label='DWA Yaw Rate')
plt.plot(hl_times, mpc_commands[:, 1], 'ro-', label='MPC Yaw Rate')
plt.title('DWA vs MPC Yaw Rate Commands')
plt.xlabel('Time (s)')
plt.ylabel('Yaw Rate (rad/s)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

total_time = timestamps[-1] if timestamps else 0
avg_speed = np.mean(trajectory[:, 3]) if len(trajectory) > 0 else 0
path_length = np.sum(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=0)) if len(trajectory) > 1 else 0
distance_to_goal = np.linalg.norm(trajectory[-1, :2] - goal_pos) if len(trajectory) > 0 else 0

print("\n=== Performance Metrics ===")
print(f"Total Time: {total_time:.2f} s")
print(f"Average Speed: {avg_speed:.3f} m/s")
print(f"Path Length: {path_length:.3f} m")
print(f"Final Distance to Goal: {distance_to_goal:.4f} m")
print(f"Computation Time: {time.time() - start_time:.2f} s")
