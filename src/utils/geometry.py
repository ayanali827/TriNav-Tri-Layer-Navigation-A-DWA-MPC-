import numpy as np
from scipy.spatial.transform import Rotation as R

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def perpendicular_distance(point, line_start, line_end):
    if np.allclose(line_start, line_end):
        return distance(point, line_start)
    return np.abs(np.cross(line_end-line_start, line_start-point)) / np.linalg.norm(line_end-line_start)

def quaternion_to_yaw(quat):
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    return rot.as_euler('zyx')[0]