import numpy as np

def to_serializable(obj):
    """
    Recursively converts numpy types to standard Python types for JSON serialization.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(v) for v in obj]
    if hasattr(obj, "to_dict"):
        return to_serializable(obj.to_dict())
    return obj
