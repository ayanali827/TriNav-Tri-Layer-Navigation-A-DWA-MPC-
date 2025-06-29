import matplotlib.pyplot as plt
import numpy as np

class SimulationVisualizer:
    def __init__(self, config):
        self.config = config
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.setup_plot()
        
        self.trajectory = []
        plt.ion()
        
    def setup_plot(self):
        self.ax.set_title('Live Trajectory Visualization')
        self.ax.set_xlabel('X Position (m)')
        self.ax.set_ylabel('Y Position (m)')
        self.ax.grid(True)
        self.ax.axis('equal')
        self.ax.set_xlim(-2, 20)
        self.ax.set_ylim(-2, 15)
        
        # Plot start and goal
        self.ax.plot(self.config.start_pos[0], self.config.start_pos[1], 'go', markersize=10, label='Start')
        self.ax.plot(self.config.goal_pos[0], self.config.goal_pos[1], 'ro', markersize=10, label='Goal')
        
        # Plot obstacles
        for obs in self.config.obstacles:
            circle = plt.Circle((obs[0], obs[1]), obs[2], color='r', alpha=0.5)
            self.ax.add_patch(circle)
        
        # Global path (initially empty)
        self.global_path_line, = self.ax.plot([], [], 'g--', alpha=0.7, label='Global Path')
        
        # Trajectory
        self.traj_line, = self.ax.plot([], [], 'b-', label='Path')
        self.dwa_traj_line, = self.ax.plot([], [], 'c--', alpha=0.5, label='DWA Plan')
        
        self.ax.legend()
        
    def update(self, state, dwa_traj=None, global_path=None):
        self.trajectory.append(state[:2])
        traj_x = [p[0] for p in self.trajectory]
        traj_y = [p[1] for p in self.trajectory]
        self.traj_line.set_data(traj_x, traj_y)
        
        if dwa_traj is not None:
            self.dwa_traj_line.set_data(dwa_traj[:,0], dwa_traj[:,1])
        
        if global_path is not None:
            self.global_path_line.set_data([p[0] for p in global_path], [p[1] for p in global_path])
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.01)
        
    def save_report(self, trajectory, controls, timestamps, dwa_commands, mpc_commands, file_prefix="report"):
        plt.ioff()
        fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        
        # Trajectory plot
        axs[0, 0].plot(trajectory[:, 0], trajectory[:, 1], 'b-', label='Actual Path')
        axs[0, 0].plot(self.config.start_pos[0], self.config.start_pos[1], 'go', markersize=10, label='Start')
        axs[0, 0].plot(self.config.goal_pos[0], self.config.goal_pos[1], 'ro', markersize=10, label='Goal')
        for obs in self.config.obstacles:
            circle = plt.Circle((obs[0], obs[1]), obs[2], color='r', alpha=0.5)
            axs[0, 0].add_patch(circle)
        axs[0, 0].set_title('Robot Trajectory with Obstacles')
        axs[0, 0].set_xlabel('X Position (m)')
        axs[0, 0].set_ylabel('Y Position (m)')
        axs[0, 0].grid(True)
        axs[0, 0].legend()
        
        # Heading
        axs[0, 1].plot(timestamps, np.degrees(trajectory[:, 2]), 'g-')
        axs[0, 1].set_title('Robot Heading')
        axs[0, 1].set_xlabel('Time (s)')
        axs[0, 1].set_ylabel('Heading (degrees)')
        axs[0, 1].grid(True)
        
        # Velocities
        axs[0, 2].plot(timestamps, trajectory[:, 3], 'b-', label='Linear Velocity')
        axs[0, 2].plot(timestamps, trajectory[:, 4], 'r-', label='Angular Velocity')
        axs[0, 2].set_title('Robot Velocities')
        axs[0, 2].set_xlabel('Time (s)')
        axs[0, 2].set_ylabel('Velocity')
        axs[0, 2].grid(True)
        axs[0, 2].legend()
        
        # Control signals
        min_length = min(len(timestamps), len(controls))
        axs[1, 0].plot(timestamps[:min_length], controls[:min_length, 0], 'b-', label='Forward Control')
        axs[1, 0].plot(timestamps[:min_length], controls[:min_length, 1], 'r-', label='Turn Control')
        axs[1, 0].set_title('Control Signals')
        axs[1, 0].set_xlabel('Time (s)')
        axs[1, 0].set_ylabel('Control Value')
        axs[1, 0].set_ylim(-1.1, 1.1)
        axs[1, 0].grid(True)
        axs[1, 0].legend()
        
        # DWA vs MPC velocity commands
        hl_times = np.array([i * self.config.high_level_freq * 0.02 for i in range(len(dwa_commands))])
        axs[1, 1].plot(hl_times, dwa_commands[:, 0], 'bo-', label='DWA Velocity')
        axs[1, 1].plot(hl_times, mpc_commands[:, 0], 'ro-', label='MPC Velocity')
        axs[1, 1].set_title('DWA vs MPC Velocity Commands')
        axs[1, 1].set_xlabel('Time (s)')
        axs[1, 1].set_ylabel('Velocity (m/s)')
        axs[1, 1].grid(True)
        axs[1, 1].legend()
        
        # DWA vs MPC yaw rate commands
        axs[1, 2].plot(hl_times, dwa_commands[:, 1], 'bo-', label='DWA Yaw Rate')
        axs[1, 2].plot(hl_times, mpc_commands[:, 1], 'ro-', label='MPC Yaw Rate')
        axs[1, 2].set_title('DWA vs MPC Yaw Rate Commands')
        axs[1, 2].set_xlabel('Time (s)')
        axs[1, 2].set_ylabel('Yaw Rate (rad/s)')
        axs[1, 2].grid(True)
        axs[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(f"{file_prefix}_summary.png")
        plt.close()