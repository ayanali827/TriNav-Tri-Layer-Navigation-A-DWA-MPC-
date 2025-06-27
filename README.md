# Tri-Nav: Hybrid Path Planning with A* + DWA + MPC

## 🚀 Goal

Develop a robust robot navigation system that combines:
- A* for global path planning,
- DWA (Dynamic Window Approach) for reactive local control,
- MPC (Model Predictive Control) for smooth trajectory optimization.

The system runs in a MuJoCo simulation environment with multiple circular obstacles.

---

## ✅ Tasks Implemented

- Grid-based global path planning using A*.
- Local dynamic window sampling and velocity selection with obstacle avoidance.
- Optimal command filtering using MPC with trajectory reference.
- PID-like motor control simulation for a differential-drive robot.
- Real-time visualization with matplotlib and MuJoCo viewer.
- Performance metrics: time, path length, speed, final error.

---

## 🧠 Design Principles

- **SOLID**: Components are modular (A*, DWA, MPC are separated).
- **DRY**: Shared logic (e.g., trajectory prediction) reused.
- **KISS**: Clear functional separation (e.g., `predict_trajectory`, `calculate_obstacle_cost`).
- **YAGNI**: No unnecessary features or abstractions.
- **Design Patterns Used**:
  - Strategy pattern (path planner selection: A*, DWA, MPC).
  - Observer-like behavior for MuJoCo simulation feedback.
  - Factory-like abstraction for planning modules.

---

## 🛠 How to Run

1. **Install MuJoCo and Dependencies**
   - Download MuJoCo and place the required `.xml` model file (DDR.xml) in your path.
   - Install dependencies:
     ```bash
     pip install -r requirements.txt
     ```

2. **Prepare the Robot Model File**
   - Ensure the path to the robot model XML is correct:
     ```python
     model = mujoco.MjModel.from_xml_path("path/to/DDR.xml")
     ```

3. **Run the Simulation**
   ```bash
   python src/tri_nav.py
