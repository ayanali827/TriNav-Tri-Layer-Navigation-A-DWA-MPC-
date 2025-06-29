import mujoco
import numpy as np
import time
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R  # ADD THIS IMPORT
from config.params import DWA_CONFIG, MPC_CONFIG, SIMULATION_PARAMS
from planning.global_planner import AStarPlanner
from planning.local_planner import dwa_planner
from control.motion_controller import mpc_controller, car_kinematic_model
from control.pid_controller import PIDController
from simulator.mujoco_simulator import get_car_state, extract_obstacles, check_collision
from utils.visualization import Visualizer
from utils.geometry import normalize_angle

def main():
    # Load parameters
    config = {**DWA_CONFIG, **MPC_CONFIG}
    sim_params = SIMULATION_PARAMS
    
    # Create path planner
    astar_planner = AStarPlanner(grid_size=0.25, robot_radius=0.3)
    global_path = astar_planner.plan_path(
        sim_params['start_pos'], 
        sim_params['goal_pos'], 
        sim_params['obstacles']
    )
    
    # Initialize simulation
    model = mujoco.MjModel.from_xml_path(r"C:\Users\ayans\Downloads\scr\scr\models\ddr.xml")
    data = mujoco.MjData(model)
    model.opt.timestep = 0.03
    
    # Set initial state
    data.qpos[0:2] = sim_params['start_pos']
    quat = R.from_euler('z', sim_params['start_yaw']).as_quat()
    data.qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]
    data.qvel[:] = 0
    
    # Initialize controllers
    v_pid = PIDController(kp=1.2, ki=0.2, kd=0.05)
    w_pid = PIDController(kp=0.8, ki=0.5, kd=0.02)
    
    # Initialize visualization
    visualizer = Visualizer(
        sim_params['obstacles'], 
        sim_params['start_pos'], 
        sim_params['goal_pos'],
        global_path
    )
    
    # Initialize data collection
    trajectory = []
    controls = []
    timestamps = []
    dwa_commands = []
    mpc_commands = []
    desired_v = 0.0
    desired_w = 0.0
    prev_desired_v = 0.0
    prev_desired_w = 0.0
    filter_alpha = 0.8
    start_time = time.time()
    
    # Initialize viewer
    try:
        from mujoco import viewer
        viewer_available = True
        viewer = viewer.launch_passive(model, data)
        viewer.sync()
    except Exception as e:
        print(f"Could not create viewer: {e}")
        viewer_available = False
    
    # Stopping controller parameters
    stopping_controller_active = False
    stopping_steps = 0
    max_stopping_steps = 20
    
    # Main simulation loop
    for step in range(sim_params['max_steps']):
        state = get_car_state(data)
        trajectory.append(state)
        timestamps.append(step * model.opt.timestep)
    
        # Calculate distance to goal
        dist_to_goal = np.linalg.norm(state[:2] - sim_params['goal_pos'])
        
        # Check if we should activate stopping controller
        if not stopping_controller_active and dist_to_goal < sim_params['stopping_threshold']:
            stopping_controller_active = True
            print(f"Activating stopping controller at step {step}, distance: {dist_to_goal:.2f}m")
        
        # Stopping controller logic
        if stopping_controller_active:
            # Set desired velocities to zero
            desired_v = 0
            desired_w = 0
            
            # Apply strong braking force
            u_forward = -1.0 * state[3]  # Braking proportional to current speed
            u_turn = -1.0 * state[4]     # Damping for angular velocity
            
            # Apply control
            data.ctrl[0] = u_forward
            data.ctrl[1] = u_turn
            controls.append([u_forward, u_turn])
            
            # Step simulation
            mujoco.mj_step(model, data)
            
            # Check if we've stopped
            current_speed = np.abs(state[3])
            current_ang_speed = np.abs(state[4])
            
            if current_speed < sim_params['stopping_speed_threshold'] and \
               current_ang_speed < sim_params['stopping_speed_threshold']:
                print(f"Stopped at goal in {step} steps! Final speed: {current_speed:.4f} m/s")
                break
                
            # Check if we've tried stopping for too long
            stopping_steps += 1
            if stopping_steps > max_stopping_steps:
                print(f"Stopping controller timed out after {max_stopping_steps} steps")
                break
                
            # Update viewer
            if viewer_available:
                viewer.sync()
                time.sleep(model.opt.timestep * 0.5)
                
            # Skip the rest of the control logic
            continue
    
        # Normal control logic (when not stopping)
        slowdown_factor = min(1.0, dist_to_goal / 1.5)
        
        # Run high-level planner at specified frequency
        if step % sim_params['high_level_freq'] == 0:
            local_config = config.copy()
            local_config['max_speed'] = config['max_speed'] * slowdown_factor
            local_config['max_yawrate'] = config['max_yawrate'] * slowdown_factor
            
            # Run DWA planner
            v_dwa, w_dwa, best_traj = dwa_planner(
                state, 
                sim_params['goal_pos'], 
                local_config, 
                sim_params['obstacles'], 
                global_path
            )
            dwa_commands.append((v_dwa, w_dwa))
            
            # Prepare reference trajectory for MPC
            if best_traj is not None:
                ref_traj = best_traj[1:1+config['horizon']]
                if len(ref_traj) < config['horizon']:
                    padding = np.tile(ref_traj[-1], (config['horizon'] - len(ref_traj), 1))
                    ref_traj = np.vstack([ref_traj, padding])
            else:
                ref_traj = np.tile(state, (config['horizon'], 1))
            
            # Run MPC controller
            desired_v, desired_w = mpc_controller(
                state, ref_traj, v_dwa, w_dwa, config, sim_params['obstacles']
            )
            mpc_commands.append((desired_v, desired_w))
            
            # Filter commands for smoothness
            desired_v = filter_alpha * prev_desired_v + (1-filter_alpha) * desired_v
            desired_w = filter_alpha * prev_desired_w + (1-filter_alpha) * desired_w
            prev_desired_v = desired_v
            prev_desired_w = desired_w
        
        # Low-level PID controller
        dt_sim = model.opt.timestep
        v_error = desired_v - state[3]
        w_error = desired_w - state[4]
        
        u_forward = v_pid.compute(v_error, dt_sim)
        u_turn = w_pid.compute(w_error, dt_sim)
        
        # Clip control signals
        u_forward = np.clip(u_forward, -1, 1)
        u_turn = np.clip(u_turn, -1, 1)
        
        # Apply control
        data.ctrl[0] = u_forward
        data.ctrl[1] = u_turn
        controls.append([u_forward, u_turn])
        
        # Simulation step
        mujoco.mj_step(model, data)
        
        # Update viewer
        if viewer_available:
            viewer.sync()
            time.sleep(model.opt.timestep * 0.5)
        
        # Update visualization
        if step % 10 == 0 and len(trajectory) > 1:
            visualizer.update(trajectory, best_traj)
    
    # Clean up viewer
    if viewer_available:
        viewer.close()
    
    # Convert data to arrays for analysis
    trajectory = np.array(trajectory)
    controls = np.array(controls)
    dwa_commands = np.array(dwa_commands)
    mpc_commands = np.array(mpc_commands)
    
    # Calculate high-level command timestamps
    hl_times = np.array([i * sim_params['high_level_freq'] * model.opt.timestep 
                        for i in range(len(dwa_commands))])
    
    # Create final plots
    visualizer.final_plot(trajectory, controls, timestamps, dwa_commands, mpc_commands, hl_times)
    
    # Performance metrics
    total_time = timestamps[-1] if timestamps else 0
    final_speed = trajectory[-1, 3] if len(trajectory) > 0 else 0
    final_ang_speed = trajectory[-1, 4] if len(trajectory) > 0 else 0
    path_length = np.sum(np.linalg.norm(np.diff(trajectory[:, :2], axis=0), axis=0)) if len(trajectory) > 1 else 0
    distance_to_goal = np.linalg.norm(trajectory[-1, :2] - sim_params['goal_pos']) if len(trajectory) > 0 else 0
    
    print("\n=== Performance Metrics ===")
    print(f"Total Time: {total_time:.2f} s")
    print(f"Final Linear Speed: {final_speed:.4f} m/s")
    print(f"Final Angular Speed: {final_ang_speed:.4f} rad/s")
    print(f"Final Distance to Goal: {distance_to_goal:.4f} m")
    print(f"Path Length: {path_length:.2f} m")
    print(f"Computation Time: {time.time() - start_time:.2f} s")
    print(f"Trajectory points: {len(trajectory)}")

if __name__ == "__main__":
    main()