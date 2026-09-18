import numpy as np

from core.data.base import Field


def magnetization(field: Field) -> np.ndarray:

    spins = field.values

    if field.batched:
        axes = tuple(range(1, spins.ndim))
        return spins.mean(axis=axes)
    else:
        return spins.mean()



def mean_magnetization(field: Field) -> float:
    if field.batched:
        return float(
            np.mean(magnetization(field))
        )
    else:
        return float(magnetization(field))


def susceptibility(field: Field, beta: float) -> float:

    m = magnetization(field)

    shape = field.values.shape[1:] if field.batched else field.values.shape
    N = np.prod(shape)

    return beta * N * np.var(m)