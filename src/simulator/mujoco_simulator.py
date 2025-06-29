import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R
from models.vehicle_state import VehicleState

class MujocoSimulator:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.collision = False

    def reset(self, start_pos, start_yaw):
        self.data.qpos[0:2] = start_pos
        quat = R.from_euler('z', start_yaw).as_quat()
        self.data.qpos[3:7] = [quat[3], quat[0], quat[1], quat[2]]
        self.data.qvel[:] = 0
        self.collision = False

    def step(self, control_signals):
        self.data.ctrl[0] = control_signals[0]
        self.data.ctrl[1] = control_signals[1]
        mujoco.mj_step(self.model, self.data)

    def get_state(self):
        return VehicleState.from_mujoco_data(self.data)

    def check_collision(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            
            if geom1_name is None or geom2_name is None:
                continue
                
            if ("chasis" in geom1_name and "obstacle" in geom2_name) or \
               ("chasis" in geom2_name and "obstacle" in geom1_name):
                self.collision = True
                return True
        return False

    def reached_goal(self, goal, threshold=0.3):
        state = self.get_state().to_array()
        return np.linalg.norm(state[:2] - goal) < threshold