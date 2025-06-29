import numpy as np

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def angle_diff(a, b):
    return normalize_angle(a - b)

def perpendicular_distance(point, line_start, line_end):
    if np.allclose(line_start, line_end):
        return np.linalg.norm(point - line_start)
    return np.abs(np.cross(line_end-line_start, line_start-point)) / np.linalg.norm(line_end-line_start)