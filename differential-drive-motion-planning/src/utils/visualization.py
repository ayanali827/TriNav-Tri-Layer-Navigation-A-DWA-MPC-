import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np

class Visualizer:
    def __init__(self, obstacles, start, goal, global_path=None):
        self.obstacles = obstacles  # Store obstacles as instance variable
        self.start = start
        self.goal = goal
        
        # Create figure and axes
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_title('Motion Planning Visualization')
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.grid(True)
        self.ax.axis('equal')
        self.ax.set_xlim(-10, 10)
        self.ax.set_ylim(-5, 15)
        
        # Draw obstacles
        for obs in self.obstacles:
            circle = Circle((obs[0], obs[1]), obs[2], color='r', alpha=0.5)
            self.ax.add_patch(circle)
        
        # Draw start and goal
        self.start_marker, = self.ax.plot([start[0]], [start[1]], 'go', markersize=10, label='Start')
        self.goal_marker, = self.ax.plot([goal[0]], [goal[1]], 'ro', markersize=10, label='Goal')
        
        # Draw global path
        if global_path is not None:
            self.global_path_line, = self.ax.plot(
                [p[0] for p in global_path], [p[1] for p in global_path], 
                'g--', alpha=0.7, label='Global Path'
            )
        
        # Initialize trajectory lines
        self.traj_line, = self.ax.plot([], [], 'b-', label='Path')
        self.dwa_traj_line, = self.ax.plot([], [], 'c--', alpha=0.5, label='DWA Plan')
        self.mpc_traj_line, = self.ax.plot([], [], 'm--', alpha=0.7, label='MPC Plan')
        
        self.ax.legend()
        plt.ion()
        self.fig.canvas.draw()
    
    def update(self, trajectory, best_traj=None, mpc_traj=None):
        # Update main trajectory
        if len(trajectory) > 0:
            traj_x = [s[0] for s in trajectory]
            traj_y = [s[1] for s in trajectory]
            self.traj_line.set_data(traj_x, traj_y)
        
        # Update DWA trajectory
        if best_traj is not None and len(best_traj) > 0:
            self.dwa_traj_line.set_data(best_traj[:,0], best_traj[:,1])
        
        # Update MPC trajectory
        if mpc_traj is not None and len(mpc_traj) > 0:
            self.mpc_traj_line.set_data(mpc_traj[:,0], mpc_traj[:,1])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def final_plot(self, trajectory, controls, timestamps, dwa_commands, mpc_commands, hl_times):
        plt.ioff()
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        
        # Trajectory plot
        axs[0, 0].plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Actual Path')
        axs[0, 0].plot(self.start[0], self.start[1], 'go', markersize=10, label='Start')
        axs[0, 0].plot(self.goal[0], self.goal[1], 'ro', markersize=10, label='Goal')
        
        # Draw obstacles
        for obs in self.obstacles:
            circle = Circle((obs[0], obs[1]), obs[2], color='r', alpha=0.5)
            axs[0, 0].add_patch(circle)
        
        axs[0, 0].set_title('Robot Trajectory')
        axs[0, 0].set_xlabel('X Position (m)')
        axs[0, 0].set_ylabel('Y Position (m)')
        axs[0, 0].grid(True)
        axs[0, 0].legend()
        
        # Heading plot
        axs[0, 1].plot(timestamps, np.degrees(trajectory[:, 2]), 'g-')
        axs[0, 1].set_title('Robot Heading')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Heading (degrees)')
        axs[0, 1].grid(True)
        
        # Velocity plot
        axs[0, 2].plot(timestamps, trajectory[:, 3], 'b-', label='Linear Velocity')
        axs[0, 2].plot(timestamps, trajectory[:, 4], 'r-', label='Angular Velocity')
        axs[0, 2].set_title('Robot Velocities')
        axs[0, 2].set_xlabel('Time (s)')
        axs[0, 2].set_ylabel('Velocity')
        axs[0, 2].grid(True)
        axs[0, 2].legend()
        
        # Control signals plot
        min_length = min(len(timestamps), len(controls))
        axs[1, 0].plot(timestamps[:min_length], controls[:min_length, 0], 'b-', label='Forward Control')
        axs[1, 0].plot(timestamps[:min_length], controls[:min_length, 1], 'r-', label='Turn Control')
        axs[1, 0].set_title('Control Signals')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('Control Value')
        axs[1, 0].set_ylim(-1.1, 1.1)
        axs[1, 0].grid(True)
        axs[1, 0].legend()
        
        # Velocity commands plot
        axs[1, 1].plot(hl_times, dwa_commands[:, 0], 'bo-', label='DWA Velocity')
        axs[1, 1].plot(hl_times, mpc_commands[:, 0], 'ro-', label='MPC Velocity')
        axs[1, 1].set_title('DWA vs MPC Velocity Commands')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('Velocity (m/s)')
        axs[1, 1].grid(True)
        axs[1, 1].legend()
        
        # Yaw rate commands plot
        axs[1, 2].plot(hl_times, dwa_commands[:, 1], 'bo-', label='DWA Yaw Rate')
        axs[1, 2].plot(hl_times, mpc_commands[:, 1], 'ro-', label='MPC Yaw Rate')
        axs[1, 2].set_title('DWA vs MPC Yaw Rate Commands')
        axs[1, 2].set_xlabel('Time (s)')
        axs[1, 2].set_ylabel('Yaw Rate (rad/s)')
        axs[1, 2].grid(True)
        axs[1, 2].legend()
        
        plt.tight_layout()
        plt.show()