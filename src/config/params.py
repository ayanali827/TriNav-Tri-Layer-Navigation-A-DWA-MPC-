import numpy as np

class SimulationParams:
    def __init__(self):
        self.model_path = "C:\\Users\\ayans\\Desktop\\Project\\Diiferential Drive Robot Model (DDR).xml"
        self.start_pos = [0, 0]
        self.start_yaw = np.pi/2
        self.goal_pos = [15, 12]
        self.max_steps = 1000
        self.goal_threshold = 0.3
        self.high_level_freq = 2
        
        self.obstacles = np.array([
            [1.2, 10.8, 0.6], [16.8, 1.2, 0.7], [2.5, 2.5, 0.5],
            [15.5, 10.5, 0.6], [1.0, 6.0, 0.7], [17.0, 6.0, 0.6],
            [4.0, 11.0, 0.5], [14.0, 1.0, 0.7], [3.0, 8.5, 0.6],
            [15.0, 3.5, 0.5], [8.0, 1.5, 0.7], [10.0, 10.5, 0.6],
            [1.5, 3.5, 0.5], [16.5, 8.5, 0.7], [5.5, 5.0, 0.6],
            [12.5, 7.0, 0.5], [9.0, 8.0, 0.7], [9.0, 4.0, 0.6],
            [3.5, 6.5, 0.5], [14.5, 5.5, 0.7], [7.0, 9.0, 0.6],
            [11.0, 3.0, 0.5], [6.0, 2.0, 0.7], [12.0, 9.0, 0.6],
            [8.5, 6.0, 0.5]
        ])
        
        self.dwa_config = {
            'max_speed': 4,
            'min_speed': -0.5,
            'max_yawrate': 10,
            'max_accel': 0.5,
            'max_dyawrate': 7,
            'dt': 0.08,
            'predict_time': 2,
            'resolution': 12,
            'weights': {'goal': 2.5, 'speed': 0.2, 'heading': 0.1, 
                        'obstacle': 0.6, 'path': 1.5},
            'robot_radius': 0.3
        }
        
        self.mpc_config = {
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
        
        self.pid_config = {
            'kp_v': 1.2,
            'ki_v': 0.2,
            'kd_v': 0.05,
            'kp_w': 0.8,
            'ki_w': 0.5,
            'kd_w': 0.02,
            'dt': 0.02
        }
        
        self.visualize = True