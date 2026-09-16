from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

from .preprocessing import (
    TensorShape,
    robust_center_scale,
    standardize,
    to_flat,
    unflatten_batch,
    unflatten_bilinear,
    unflatten_vector,
)


@dataclass
class LocalEstimate:
    x: object
    n_neighbors: int
    n_continuous: int
    drift: object | None
    diffusion: object | None
    n_jumps_nearby: int
    nearest_neighbor_distance: float


@dataclass
class SDEEstimationResult:
    tensor_shape: list[int]
    global_drift: object | None
    global_diffusion: object | None
    n_jumps: int
    global_jump_intensity: float | None
    global_jump_mean: object | None
    global_jump_covariance: object | None
    jump_detected: bool
    model_description: str
    local_estimates: list[LocalEstimate] = field(default_factory=list)
    warning: str | None = None


def _cumulants(dX: np.ndarray, mask: np.ndarray):
    """Mean per-step drift vector and Var[dX] diffusion matrix over `mask`
    (flat representation; dX is already per-step, no dt normalization)."""
    if not np.any(mask):
        return None, None
    dX_m = dX[mask]
    drift = np.mean(dX_m, axis=0)
    dX_m = dX_m - drift  # center for diffusion estimate
    outer = np.einsum("ni,nj->nij", dX_m, dX_m)
    diffusion = np.mean(outer, axis=0)
    return drift, diffusion


def _detect_jumps(dX: np.ndarray, jump_zscore: float) -> np.ndarray:
    n, D = dX.shape
    if n < max(4, D + 2):
        return np.zeros(n, dtype=bool)

    center, scale = robust_center_scale(dX)
    z = standardize(dX, center, scale)
    # If no jump then, z is supposed to be a D dimensional vector with 
    # standard normal distribution. The squared robust Mahalanobis distance 
    # is then approximately a chi-squared distribution with D degrees of freedom. 
    # Here we detect jumps by checking if the squared robust Mahalanobis distance 
    # has a heavy tail. In other words, we check if there are points which, 
    # roughly speaking, have a value that exceeds the +-jump_zscore * variance 
    # in some directions.
    mahalanobis_sq = np.sum(z ** 2, axis=1)
    two_sided_tail_prob = 2.0 * stats.norm.sf(jump_zscore)
    threshold = stats.chi2.ppf(1.0 - two_sided_tail_prob, df=D)
    return mahalanobis_sq > threshold


def _order_by_time(X: np.ndarray, t: np.ndarray | None) -> np.ndarray:
    if t is None:
        return X
    order = np.argsort(np.asarray(t, dtype=float))
    return X[order]


class LocalSDEModel:
    """A local regression model of mu(x) and Sigma(x), queryable at any x."""

    def __init__(
        self,
        x_start: np.ndarray,
        dX: np.ndarray,
        is_jump: np.ndarray,
        k: int,
        tensor_shape: TensorShape,
    ):
        self.dX = dX
        self.is_jump = is_jump
        self.cont_mask = ~is_jump
        self.n = len(dX)
        self.k = min(k, self.n)
        self.tensor_shape = tensor_shape
        self.center, self.scale = robust_center_scale(x_start)
        self.z_start = standardize(x_start, self.center, self.scale)
        self._tree = cKDTree(self.z_start)

    def _coerce_query(self, x_query) -> np.ndarray:
        """Resolve a query into a flat (m, D) batch, given this model's known
        tensor_shape. If x_query is a single state, returns (1, D); if it's a batch of states, returns (m, D). Raises ValueError if the shape is incompatible."""
        arr = np.asarray(x_query, dtype=float)
        ts = self.tensor_shape
        if arr.shape == ts:
            return arr.reshape(1, -1)  # exactly one tensor state
        if arr.ndim == len(ts) + 1 and arr.shape[1:] == ts:
            return arr.reshape(arr.shape[0], -1)  # a batch of tensor states
        raise ValueError(
            f"expected a single state of shape {ts} or a batch of shape (m,) + {ts}, got {arr.shape}"
        )

    def predict_kNN(self, x_query) -> list[LocalEstimate]:
        """Estimate local drift/diffusion/jump activity at one or more query
        states using k-NN regression. `x_query` may be a single state (matching this model's
        tensor shape) or a batch of them; it need not be a state that was
        ever observed."""
        Xq = self._coerce_query(x_query)
        Zq = standardize(Xq, self.center, self.scale)
        dist, idx = self._tree.query(Zq, k=self.k)
        dist = np.atleast_2d(dist)
        idx = np.atleast_2d(idx)

        results = []
        for row in range(len(Xq)):
            neighbor_idx = np.atleast_1d(idx[row])
            mask = np.zeros(self.n, dtype=bool)
            mask[neighbor_idx] = True
            cont = mask & self.cont_mask

            drift, diffusion = _cumulants(self.dX, cont)
            results.append(
                LocalEstimate(
                    x=unflatten_vector(Xq[row], self.tensor_shape).tolist(),
                    n_neighbors=int(len(neighbor_idx)),
                    n_continuous=int(np.sum(cont)),
                    drift=(unflatten_vector(drift, self.tensor_shape).tolist() if drift is not None else None),
                    diffusion=(
                        unflatten_bilinear(diffusion, self.tensor_shape).tolist()
                        if diffusion is not None
                        else None
                    ),
                    n_jumps_nearby=int(np.sum(mask & self.is_jump)),
                    nearest_neighbor_distance=float(np.min(dist[row])),
                )
            )
        return results

    def predict_local_linear_regression(self, x_query) -> list[LocalEstimate]:
        Xq = self._coerce_query(x_query)
        Zq = standardize(Xq, self.center, self.scale)
        dist, idx = self._tree.query(Zq, k=self.k)
        dist = np.atleast_2d(dist)
        idx = np.atleast_2d(idx)

        true_indices = np.where(self.cont_mask)[0]

        results = []
        for row in range(len(Xq)):
            neighbor_idx = np.atleast_1d(idx[row])
            mask = np.zeros(self.n, dtype=bool)
            mask[neighbor_idx] = True
            cont = mask & self.cont_mask

            kk = int(np.sum(cont))
            filtered_dist = dist[row, np.isin(idx[row], true_indices)]  # only consider continuous neighbors

            h = filtered_dist[-1]  # bandwidth for each query point
            u = filtered_dist / h  # normalized distances
            w = (1 - u**3) ** 3  # tricube kernel weights

            diff = self.z_start[cont] - Zq[row]

            W = np.diag(w)

            design_mat = np.hstack([np.ones((kk, 1)), diff])

            beta = np.linalg.solve(design_mat.T @ W @ design_mat, design_mat.T @ W @ self.dX[cont])

            drift = beta[0]

            residuals = self.dX[cont] - design_mat @ beta
            log_res_sq = np.log(residuals**2 + 1e-12)  # add small constant to avoid log(0)
            beta_log = np.linalg.solve(design_mat.T @ W @ design_mat, design_mat.T @ W @ log_res_sq)

            log_s2_corrected = beta_log[0] - 1.27036  # bias correction for log of squared residuals
            diffusion = np.exp(log_s2_corrected)
            
            results.append(
                LocalEstimate(
                    x=unflatten_vector(Xq[row], self.tensor_shape).tolist(),
                    n_neighbors=int(len(neighbor_idx)),
                    n_continuous=int(np.sum(cont)),
                    drift=(unflatten_vector(drift, self.tensor_shape).tolist() if drift is not None else None),
                    diffusion=(
                        unflatten_bilinear(diffusion, self.tensor_shape).tolist()
                        if diffusion is not None
                        else None
                    ),
                    n_jumps_nearby=int(np.sum(mask & self.is_jump)),
                    nearest_neighbor_distance=float(np.min(dist[row])),
                )
            )

        return results


def fit_local_sde_model(
    x,
    t=None,
    k: int | None = None,
    jump_zscore: float = 3.5,
) -> LocalSDEModel | None:
    """Fit a queryable mu(x)/Sigma(x) model from a discrete-time series of
    tensors. Returns None if there isn't enough data (fewer than 2 usable
    increments)."""
    X, tensor_shape = to_flat(x)
    X = _order_by_time(X, t)

    dX = np.diff(X, axis=0)
    x_start = X[:-1]
    n = len(dX)
    if n < 2:
        return None

    is_jump = _detect_jumps(dX, jump_zscore)
    k_local = k if k is not None else max(2, min(8, int(np.sqrt(n))))
    return LocalSDEModel(x_start, dX, is_jump, k_local, tensor_shape)


def estimate_sde_terms(
    x,
    t=None,
    k: int | None = None,
    jump_zscore: float = 3.5,
    n_query_points: int = 6,
    query_states=None,
) -> SDEEstimationResult:
    """Summarize the fitted jump-diffusion model: a global (homogeneous) fit,
    plus local drift/diffusion/jump estimates at a handful of states."""
    X, tensor_shape = to_flat(x)
    X = _order_by_time(X, t)

    dX = np.diff(X, axis=0)
    x_start = X[:-1]
    n = len(dX)

    if n == 0:
        return SDEEstimationResult(
            tensor_shape=list(tensor_shape),
            global_drift=None,
            global_diffusion=None,
            n_jumps=0,
            global_jump_intensity=None,
            global_jump_mean=None,
            global_jump_covariance=None,
            jump_detected=False,
            model_description="insufficient data",
            warning="need at least two time steps to estimate anything",
        )

    is_jump = _detect_jumps(dX, jump_zscore)
    cont_mask = ~is_jump
    n_jumps = int(np.sum(is_jump))

    global_drift, global_diffusion = _cumulants(dX, cont_mask)

    if n_jumps > 0:
        global_jump_intensity = n_jumps / n
        global_jump_mean = np.mean(dX[is_jump], axis=0)
        D = dX.shape[1]
        global_jump_covariance = (
            np.cov(dX[is_jump], rowvar=False).reshape(D, D) if n_jumps > 1 else np.zeros((D, D))
        )
    else:
        global_jump_intensity = 0.0
        global_jump_mean = None
        global_jump_covariance = None

    jump_detected = n_jumps > 0
    model_description = (
        "jump-diffusion (drift + diffusion + jumps)"
        if jump_detected
        else "pure diffusion (drift + diffusion, no jumps detected)"
    )

    warning = None
    if n < 4:
        warning = (
            "series too short for reliable jump detection (need >= 4 increments); "
            "all increments were treated as continuous."
        )

    local_estimates: list[LocalEstimate] = []
    if n >= 2:
        k_local = k if k is not None else max(2, min(8, int(np.sqrt(n))))
        model = LocalSDEModel(x_start, dX, is_jump, k_local, tensor_shape)

        if query_states is not None:
            query_x = query_states
        else:
            n_q = max(1, min(n_query_points, n))
            idxs = sorted(set(np.linspace(0, n - 1, n_q).astype(int).tolist()))
            query_x = unflatten_batch(x_start[idxs], tensor_shape)

        local_estimates = model.predict_kNN(query_x)

    return SDEEstimationResult(
        tensor_shape=list(tensor_shape),
        global_drift=unflatten_vector(global_drift, tensor_shape).tolist() if global_drift is not None else None,
        global_diffusion=(
            unflatten_bilinear(global_diffusion, tensor_shape).tolist() if global_diffusion is not None else None
        ),
        n_jumps=n_jumps,
        global_jump_intensity=global_jump_intensity,
        global_jump_mean=(
            unflatten_vector(global_jump_mean, tensor_shape).tolist() if global_jump_mean is not None else None
        ),
        global_jump_covariance=(
            unflatten_bilinear(global_jump_covariance, tensor_shape).tolist()
            if global_jump_covariance is not None
            else None
        ),
        jump_detected=jump_detected,
        model_description=model_description,
        local_estimates=local_estimates,
        warning=warning,
    )
