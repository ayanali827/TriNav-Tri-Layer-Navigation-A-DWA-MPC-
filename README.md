# Motion Planning for Differential Drive Robots

## Introduction

Motion planning involves computing the state sequence for a robot to move from start to goal without conflicts. This repository implements a comprehensive motion planning system combining:

- **Path Planning**: Computes optimal collision-free paths considering obstacles
- **Trajectory Planning**: Generates motion states based on kinematics and dynamics constraints
- **Control**: Executes planned trajectories using advanced controllers

The system uses a hierarchical approach:
1. Global path planning with A* algorithm
2. Local trajectory optimization with Dynamic Window Approach (DWA)
3. Model Predictive Control (MPC) for precise trajectory following
4. PID for low-level motor control

## Key Features

- **Modular Architecture**: SOLID-compliant design with clear separation of concerns
- **Multiple Planning Strategies**: Combines global A* with local DWA and MPC
- **Collision Avoidance**: Real-time obstacle detection and avoidance
- **Visualization**: Comprehensive trajectory visualization and performance metrics
- **Smooth Control**: PID controller with velocity filtering for smooth operation

## Algorithms Implemented

| Category         | Algorithm      | Status | Animation Example |
|------------------|----------------|--------|-------------------|
| Global Planning  | A*             | ✅     | ![A* Path](docs/a_star_path.png) |
| Local Planning   | DWA            | ✅     | ![DWA Planning](docs/dwa_planning.gif) |
| Trajectory Opt.  | MPC            | ✅     | ![MPC Control](docs/mpc_control.gif) |
| Control          | PID            | ✅     | ![PID Control](docs/pid_control.png) |

## Installation

### Prerequisites
- Python 3.10+
- MuJoCo (with valid license)

### Using pip
```bash
pip install numpy scipy matplotlib mujoco
```

### From Source
```bash
git clone https://github.com/yourusername/differential-drive-motion-planning.git
cd differential-drive-motion-planning
pip install -r requirements.txt
```

## Project Structure

```
differential-drive-motion-planning/
├── models/                  # Robot models and environments
│   └── ddr.xml              # Differential drive robot MuJoCo model
├── src/                     # Main source code
│   ├── config/              # Configuration parameters
│   │   └── params.py        # Simulation parameters
│   ├── control/             # Control algorithms
│   │   ├── motion_controller.py  # MPC controller
│   │   └── pid_controller.py     # PID controller
│   ├── models/              # Data models
│   │   ├── environment.py   # Environment representation
│   │   └── vehicle_state.py # Vehicle state model
│   ├── planning/            # Planning algorithms
│   │   ├── global_planner.py# A* path planner
│   │   └── local_planner.py # DWA trajectory planner
│   ├── simulator/           # Simulation components
│   │   ├── base_simulator.py# Simulator interface
│   │   └── mujoco_simulator.py  # MuJoCo implementation
│   ├── utils/               # Utility functions
│   │   ├── geometry.py      # Geometry calculations
│   │   └── visualization.py # Visualization tools
│   └── main.py              # Main entry point
├── requirements.txt         # Python dependencies
└── README.md                # This document
```

## Usage

### Basic Simulation
```bash
python src/main.py
```

### Customizing Parameters
Modify `src/config/params.py` to:
- Change start/goal positions
- Adjust obstacle configurations
- Tune planning and control parameters

### Example Configuration
```python
# In src/config/params.py
start_pos = [0, 0]           # Starting position [x, y]
goal_pos = [15, 12]          # Goal position [x, y]
obstacles = [                # List of obstacles [x, y, radius]
    [1.2, 10.8, 0.6],
    [16.8, 1.2, 0.7],
    # ... add more obstacles
]
max_speed = 4.0              # Maximum robot speed (m/s)
```

## Implemented Functions

### Planning
- **A* Global Planner**: Computes optimal path using grid-based search
- **Dynamic Window Approach**: Local trajectory optimization with obstacle avoidance
- **Path Simplification**: Reduces path complexity while maintaining safety

### Control
- **Model Predictive Control**: Optimizes trajectory following
- **PID Controller**: Executes velocity commands with smooth transitions
- **Recovery Behaviors**: Handles dead-end situations

### Simulation
- **MuJoCo Integration**: Realistic physics simulation
- **Collision Detection**: Continuous collision checking
- **State Estimation**: Accurate pose and velocity tracking

### Visualization
- **Live Trajectory Plotting**: Real-time path visualization
- **Performance Metrics**: Path length, average speed, computation time
- **Command Comparison**: DWA vs MPC commands visualization

## Examples of Work

### Navigation Through Obstacles
![Obstacle Navigation](docs/obstacle_navigation.gif)

### Performance Metrics
![Performance Report](docs/performance_report.png)

### Control Signals
![Control Signals](docs/control_signals.png)

## Design Patterns and Principles

### SOLID Principles
- **Single Responsibility**: Each class has a single purpose
- **Open/Closed**: Extensible through interfaces and inheritance
- **Liskov Substitution**: Interchangeable components
- **Interface Segregation**: Focused, minimal interfaces
- **Dependency Inversion**: High-level modules depend on abstractions

### Design Patterns
- **Strategy**: Interchangeable planning and control algorithms
- **Factory**: Creates different planner types
- **Observer**: Visualization updates on state changes
- **Facade**: Simplified interfaces for complex subsystems

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature`)
3. Commit your changes (`git commit -am 'Add some feature`)
4. Push to the branch (`git push origin feature/your-feature`)
5. Create a Pull Request

## Acknowledgment

This project references and builds upon concepts from:
- [Python Motion Planning](https://github.com/zhm-real/PathPlanning)
- [MuJoCo Physics Simulator](https://mujoco.org/)
- [Dynamic Window Approach](https://www.ri.cmu.edu/pub_files/pub1/fox_dieter_1997_1/fox_dieter_1997_1.pdf)
- [Model Predictive Control](https://arxiv.org/abs/1705.02789)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
