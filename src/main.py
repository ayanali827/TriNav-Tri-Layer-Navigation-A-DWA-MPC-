import numpy as np
import time
import mujoco
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from config.params import SimulationParams
from simulator.mujoco_simulator import MujocoSimulator
from planning.global_planner import AStarPlanner
from planning.local_planner import DWAPlanner
from control.motion_controller import MPCController
from control.pid_controller import PIDController
from utils.visualization import SimulationVisualizer
from models.environment import Environment

def main():
    config = SimulationParams()
    
    # Create simulator
    simulator = MujocoSimulator(config.model_path)
    simulator.reset(config.start_pos, config.start_yaw)
    
    # Create environment
    env = Environment(simulator.model, simulator.data)
    env.extract_obstacles()
    
    # Create planners and controllers
    global_planner = AStarPlanner(grid_size=0.3, robot_radius=0.4)
    global_path = global_planner.plan_path(config.start_pos, config.goal_pos, env.obstacles)
    
    local_planner = DWAPlanner(config.dwa_config)
    mpc_controller = MPCController(config.mpc_config)
    pid_controller = PIDController(config.pid_config)
    
    # Visualization
    visualizer = SimulationVisualizer(config) if config.visualize else None
    
    # Data recording
    trajectory = []
    controls = []
    timestamps = []
    dwa_commands = []
    mpc_commands = []
    
    start_time = time.time()
    
    for step in range(config.max_steps):
        state = simulator.get_state().to_array()
        trajectory.append(state)
        timestamps.append(step * simulator.model.opt.timestep)
        
        dist_to_goal = np.linalg.norm(state[:2] - config.goal_pos)
        if dist_to_goal < config.goal_threshold:
            print(f"Goal reached in {step} steps!")
            break
        
        # Global planning (less frequent)
        if step % config.high_level_freq == 0:
            global_path = global_planner.plan_path(state[:2], config.goal_pos, env.obstacles)
            if global_path is None:
                print("Global planning failed, using previous path")
        
        # Local planning with DWA
        v_dwa, w_dwa, best_traj = local_planner.plan(state, config.goal_pos, env.obstacles, global_path)
        dwa_commands.append((v_dwa, w_dwa))
        
        # MPC optimization
        desired_v, desired_w = mpc_controller.optimize(state, best_traj, v_dwa, w_dwa, env.obstacles)
        mpc_commands.append((desired_v, desired_w))
        
        # Low-level PID control
        control_signals = pid_controller.compute_control(state, desired_v, desired_w)
        controls.append(control_signals)
        
        # Simulation step
        simulator.step(control_signals)
        
        # Check collision
        if simulator.check_collision():
            print("Collision detected! Stopping simulation.")
            break
        
        # Visualization update
        if visualizer is not None:
            visualizer.update(state, best_traj, global_path)
    
    # Convert to arrays
    trajectory = np.array(trajectory)
    controls = np.array(controls)
    dwa_commands = np.array(dwa_commands)
    mpc_commands = np.array(mpc_commands)
    
    # Save report
    if visualizer is not None:
        visualizer.save_report(trajectory, controls, timestamps, dwa_commands, mpc_commands)
    
    # Performance metrics
    total_time = timestamps[-1] if timestamps else 0
    avg_speed = np.mean(trajectory[:, 3]) if len(trajectory) > 0 else 0
    path_length = np.sum(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=1)) if len(trajectory) > 1 else 0
    distance_to_goal = np.linalg.norm(trajectory[-1, :2] - config.goal_pos) if len(trajectory) > 0 else 0

    print("\n=== Performance Metrics ===")
    print(f"Total Time: {total_time:.2f} s")
    print(f"Average Speed: {avg_speed:.3f} m/s")
    print(f"Path Length: {path_length:.3f} m")
    print(f"Final Distance to Goal: {distance_to_goal:.4f} m")
    print(f"Computation Time: {time.time() - start_time:.2f} s")
    
    if visualizer is not None:
        plt.show()

if __name__ == "__main__":
    main()