# utils.py
import numpy as np

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    else:
        return obj

def sort_vertices_counterclockwise(vertices):
    center = np.mean(vertices, axis=0)
    return sorted(vertices, key=lambda v: np.arctan2(v[1] - center[1], v[0] - center[0]))
