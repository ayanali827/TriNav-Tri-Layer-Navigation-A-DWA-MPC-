import numpy as np
from scipy.optimize import minimize
from utils.geometry import normalize_angle

class MPCController:
    def __init__(self, config):
        self.config = config
        
    def car_kinematic_model(self, state, v_cmd, w_cmd, dt):
        x, y, yaw, v_curr, w_curr = state
        tau_v = 0.2
        tau_w = 0.1
        
        v_next = v_curr + (v_cmd - v_curr) * (dt / tau_v)
        w_next = w_curr + (w_cmd - w_curr) * (dt / tau_w)
        
        x_new = x + v_next * np.cos(yaw) * dt
        y_new = y + v_next * np.sin(yaw) * dt
        yaw_new = yaw + w_next * dt
        
        return np.array([x_new, y_new, normalize_angle(yaw_new), v_next, w_next])

    def calculate_obstacle_cost(self, state, obstacles, robot_radius):
        pos = state[:2]
        distances = [np.linalg.norm(pos - obstacle[:2]) for obstacle in obstacles]
        min_dist = min(distances) if distances else float('inf')
        safety_margin = robot_radius + 0.1
        
        if min_dist < safety_margin:
            return float('inf')
        elif min_dist < safety_margin + 0.5:
            return 1.0 / (min_dist - safety_margin + 1e-6)
        else:
            return 0.0

    def optimize(self, current_state, ref_trajectory, v_dwa, w_dwa, obstacles):
        cfg = self.config
        if len(ref_trajectory) < cfg['horizon']:
            last_state = ref_trajectory[-1] if len(ref_trajectory) > 0 else current_state
            padding = np.tile(last_state, (cfg['horizon'] - len(ref_trajectory), 1))
            ref_traj = np.vstack([ref_trajectory, padding])
        else:
            ref_traj = ref_trajectory[:cfg['horizon']]
        
        def cost_function(u):
            v_cmds = u[:cfg['horizon']]
            w_cmds = u[cfg['horizon']:]
            cost = 0.0
            state = current_state.copy()
            
            for t in range(cfg['horizon']):
                state = self.car_kinematic_model(state, v_cmds[t], w_cmds[t], cfg['dt_mpc'])
                ref_state = ref_traj[t]
                
                pos_error = np.linalg.norm(state[:2] - ref_state[:2])
                cost += cfg['weights_mpc']['position'] * pos_error
                
                heading_error = abs(normalize_angle(state[2] - ref_state[2]))
                cost += cfg['weights_mpc']['heading'] * heading_error
                
                obstacle_cost = self.calculate_obstacle_cost(state, obstacles, 0.3)
                cost += cfg['weights_mpc']['obstacle'] * obstacle_cost
                
                if t > 0:
                    dv = v_cmds[t] - v_cmds[t-1]
                    dw = w_cmds[t] - w_cmds[t-1]
                    cost += cfg['weights_mpc']['dv'] * dv**2 + cfg['weights_mpc']['dw'] * dw**2
                
                cost += cfg['weights_mpc']['v'] * (v_cmds[t] - v_dwa)**2
                cost += cfg['weights_mpc']['w'] * (w_cmds[t] - w_dwa)**2
            
            return cost
        
        v_bounds = [(cfg['min_speed'], cfg['max_speed'])] * cfg['horizon']
        w_bounds = [(-cfg['max_yawrate'], cfg['max_yawrate'])] * cfg['horizon']
        bounds = v_bounds + w_bounds
        
        u0 = np.concatenate([np.ones(cfg['horizon']) * v_dwa, 
                             np.ones(cfg['horizon']) * w_dwa])
        
        res = minimize(cost_function, u0, method='SLSQP', bounds=bounds, 
                       options={'maxiter': 50, 'ftol': 1e-4})
        
        if not res.success:
            return v_dwa, w_dwa
        
        v_opt = res.x[0]
        w_opt = res.x[cfg['horizon']]
        return v_opt, w_opt