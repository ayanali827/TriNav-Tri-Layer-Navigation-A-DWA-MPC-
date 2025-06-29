import mujoco
import numpy as np
import math

class Environment:
    def __init__(self, model, data):
        self.model = model
        self.data = data
        self.obstacles = []

    def extract_obstacles(self):
        self.obstacles = []
        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name and "obstacle" in geom_name:
                pos = self.model.geom_pos[i]
                size = self.model.geom_size[i]
                circumradius = math.sqrt(size[0]**2 + size[1]**2)
                self.obstacles.append([pos[0], pos[1], circumradius])
        return np.array(self.obstacles)

    def check_collision(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1)
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2)
            
            if geom1_name is None or geom2_name is None:
                continue
                
            if ("chasis" in geom1_name and "obstacle" in geom2_name) or \
               ("chasis" in geom2_name and "obstacle" in geom1_name):
                return True
        return False