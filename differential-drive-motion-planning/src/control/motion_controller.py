import numpy as np
from scipy.optimize import minimize
from utils.geometry import normalize_angle

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
    
    # Prepare reference trajectory
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
            
            # Position error cost
            pos_error = np.linalg.norm(state[:2] - ref_state[:2])
            cost += weights['position'] * pos_error
            
            # Heading error cost
            heading_error = abs(normalize_angle(state[2] - ref_state[2]))
            cost += weights['heading'] * heading_error
            
            # Obstacle avoidance cost
            obstacle_cost = calculate_obstacle_cost([state], obstacles, robot_radius)
            # Cap obstacle cost to avoid infinity
            if obstacle_cost == float('inf'):
                obstacle_cost = 1e6
            cost += weights['obstacle'] * obstacle_cost
            
            # Control smoothness costs
            if t > 0:
                dv = abs(v_cmds[t] - v_cmds[t-1]) / dt_mpc
                dw = abs(w_cmds[t] - w_cmds[t-1]) / dt_mpc
                cost += weights['dv'] * min(dv, 5.0)
                cost += weights['dw'] * min(dw, 10.0)
            
            # Tracking of DWA commands
            cost += weights['v'] * (v_cmds[t] - v_dwa)**2
            cost += weights['w'] * (w_cmds[t] - w_dwa)**2
        
        return cost
    
    # Set bounds for optimization
    v_bounds = [(config['min_speed'], config['max_speed'])] * horizon
    w_bounds = [(-config['max_yawrate'], config['max_yawrate'])] * horizon
    bounds = v_bounds + w_bounds
    
    # Initial guess (DWA commands for whole horizon)
    u0 = np.concatenate([np.ones(horizon) * v_dwa, 
                         np.ones(horizon) * w_dwa])
    
    # Clip initial guess to bounds
    u0_clipped = np.clip(u0, 
                         [b[0] for b in bounds], 
                         [b[1] for b in bounds])
    
    # Run optimization with increased iterations
    res = minimize(cost_function, u0_clipped, method='SLSQP', bounds=bounds, 
                   options={'maxiter': 100, 'ftol': 1e-3})
    
    # Always use result if available, even if not fully converged
    if res.status >= 0:
        v_opt = res.x[0]
        w_opt = res.x[horizon]
    else:
        # If optimization failed completely, fall back to DWA commands
        v_opt, w_opt = v_dwa, w_dwa
    
    return v_opt, w_opt

# Helper function from local_planner needed for MPC
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