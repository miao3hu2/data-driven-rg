from __future__ import annotations

import numpy as np

TensorShape = tuple[int, ...]


def to_flat(x) -> tuple[np.ndarray, TensorShape]:
    """Flatten a time series of tensors, shape (n, *tensor_shape), into a
    computation-friendly (n, D) array, D = prod(tensor_shape)."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        raise ValueError("expected a time series (at least one leading time axis), got a single scalar")
    n = arr.shape[0]
    tensor_shape = arr.shape[1:]
    flat = arr.reshape(n, -1)
    return flat, tensor_shape


def unflatten_vector(v: np.ndarray, tensor_shape: TensorShape):
    """Reshape one flat length-D vector back into tensor_shape."""
    return v.reshape(tensor_shape)


def unflatten_batch(V: np.ndarray, tensor_shape: TensorShape) -> np.ndarray:
    """Reshape a batch (m, D) back into (m, *tensor_shape)."""
    return V.reshape((V.shape[0],) + tensor_shape)


def unflatten_bilinear(M: np.ndarray, tensor_shape: TensorShape) -> np.ndarray:
    """Reshape a flat (D, D) second-moment/covariance matrix into a
    (*tensor_shape, *tensor_shape) tensor -- the natural generalization of a
    covariance matrix to tensor-valued data: Sigma[i1..ik, j1..jk] =
    E[dX[i1..ik] dX[j1..jk]]."""
    return M.reshape(tensor_shape + tensor_shape)


def robust_center_scale(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-(flattened-)dimension median/MAD center and scale, falling back
    to std for dimensions with zero MAD (e.g. many repeated values). `X` is
    already flat, shape (n, D)."""
    median = np.median(X, axis=0)
    mad = np.median(np.abs(X - median), axis=0)
    scale = 1.4826 * mad

    flat = scale < 1e-12
    if np.any(flat):
        scale[flat] = np.std(X[:, flat], axis=0)
    scale[scale < 1e-12] = 1.0
    return median, scale


def standardize(X: np.ndarray, center: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (X - center) / scale
