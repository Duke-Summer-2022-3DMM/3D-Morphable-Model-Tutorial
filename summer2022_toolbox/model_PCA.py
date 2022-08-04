import numpy as np
from tqdm import tqdm

def single_flat_to_pointcloud(flat_shape):
    """
    convert point cloud from shape (3n, 1) to shape (3, n)
    """
    return flat_shape.reshape(((int)(flat_shape.shape[0] / 3), 3)).T

def pointcloud_to_flat(X_all):
    X_flatten = []
    for X in X_all:
        X_flatten.append(np.asarray(X).flatten('F'))

    return np.asarray(X_flatten)

def flat_to_pointcloud(X_flat_all):
    n_point = (int)(X_flat_all.shape[1] / 3)
    X_recon_all = []
    for X in X_flat_all:
        X_recon_all.append(X.reshape(n_point, 3).T)

    return np.asarray(X_recon_all)