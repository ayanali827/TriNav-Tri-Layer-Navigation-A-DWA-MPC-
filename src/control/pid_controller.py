import numpy as np

class PIDController:
    def __init__(self, config):
        self.config = config
        self.prev_v_error = 0
        self.integral_v = 0
        self.prev_w_error = 0
        self.integral_w = 0
        self.prev_desired_v = 0
        self.prev_desired_w = 0
        self.filter_alpha = 0.8

    def compute_control(self, current_state, desired_v, desired_w):
        # Low-pass filter for desired commands
        desired_v = self.filter_alpha * self.prev_desired_v + (1 - self.filter_alpha) * desired_v
        desired_w = self.filter_alpha * self.prev_desired_w + (1 - self.filter_alpha) * desired_w
        self.prev_desired_v = desired_v
        self.prev_desired_w = desired_w

        # PID for velocity
        v_error = desired_v - current_state[3]
        w_error = desired_w - current_state[4]

        self.integral_v += v_error * self.config['dt']
        derivative_v = (v_error - self.prev_v_error) / self.config['dt']
        u_forward = (self.config['kp_v'] * v_error +
                     self.config['ki_v'] * self.integral_v +
                     self.config['kd_v'] * derivative_v)

        self.integral_w += w_error * self.config['dt']
        derivative_w = (w_error - self.prev_w_error) / self.config['dt']
        u_turn = (self.config['kp_w'] * w_error +
                  self.config['ki_w'] * self.integral_w +
                  self.config['kd_w'] * derivative_w)

        self.prev_v_error = v_error
        self.prev_w_error = w_error

        # Clamp control outputs
        u_forward = np.clip(u_forward, -1, 1)
        u_turn = np.clip(u_turn, -1, 1)

        return u_forward, u_turn