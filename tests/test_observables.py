import numpy as np

from core.data.base import Field
from evaluation.observables import magnetization, mean_magnetization, susceptibility


def test_magnetization_non_batched():
    field = Field(values=np.array([[1, 1], [-1, 1]]), scale=1.0)
    assert magnetization(field) == 0.5


def test_magnetization_batched_returns_per_sample_mean():
    values = np.array(
        [
            [[1, 1], [1, 1]],   # mean = 1
            [[-1, -1], [-1, -1]],  # mean = -1
            [[1, -1], [-1, 1]],  # mean = 0
        ]
    )
    field = Field(values=values, scale=1.0, batched=True)
    np.testing.assert_allclose(magnetization(field), [1.0, -1.0, 0.0])


def test_mean_magnetization_non_batched():
    field = Field(values=np.array([[1, 1], [1, -1]]), scale=1.0)
    assert mean_magnetization(field) == 0.5


def test_mean_magnetization_batched_averages_across_samples():
    values = np.array(
        [
            [[1, 1], [1, 1]],   # mean = 1
            [[-1, -1], [-1, -1]],  # mean = -1
        ]
    )
    field = Field(values=values, scale=1.0, batched=True)
    assert mean_magnetization(field) == 0.0


def test_susceptibility_known_variance():
    # Two samples with magnetizations +1 and -1 -> var(m) = 1.
    values = np.array(
        [
            np.ones((2, 2)),
            -np.ones((2, 2)),
        ]
    )
    field = Field(values=values, scale=1.0, batched=True)
    beta = 1.0
    N = 4  # spatial size per sample

    assert susceptibility(field, beta) == beta * N * 1.0


def test_susceptibility_zero_for_constant_magnetization():
    values = np.array(
        [
            np.ones((2, 2)),
            np.ones((2, 2)),
        ]
    )
    field = Field(values=values, scale=1.0, batched=True)
    assert susceptibility(field, beta=2.0) == 0.0
