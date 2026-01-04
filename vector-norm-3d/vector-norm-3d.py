import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    norm = np.sqrt(np.square(v).sum(axis=-1))

    return norm