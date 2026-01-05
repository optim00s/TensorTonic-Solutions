import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    x_np = np.array(x)
    y_np = np.array(y)

    if x_np.ndim != 1 or y_np.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")

    if x_np.shape != y_np.shape:
        raise ValueError("Arrays must have the same length")
    
    dot_product = np.dot(x_np, y_np)

    return float(dot_product)

    