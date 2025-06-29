import numpy as np
from utils.geometry import quaternion_to_yaw

class VehicleState:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, vx_body=0.0, wz=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.vx_body = vx_body
        self.wz = wz

    def to_array(self):
        return np.array([self.x, self.y, self.yaw, self.vx_body, self.wz])

    @classmethod
    def from_mujoco_data(cls, data):
        x = data.qpos[0]
        y = data.qpos[1]
        quat = data.qpos[3:7]
        yaw = quaternion_to_yaw(quat)
        
        vx_global = data.qvel[0]
        vy_global = data.qvel[1]
        vx_body = vx_global * np.cos(yaw) + vy_global * np.sin(yaw)
        wz = data.qvel[5]
        
        return cls(x, y, yaw, vx_body, wz)