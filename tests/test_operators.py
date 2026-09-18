import numpy as np
import pytest

from core.data.base import Field
from core.operators.base import Operator, OperatorBasis
from core.operators.ising import NearestNeighbor, NextNearestNeighbor


# 3x3 periodic spin configuration used across tests. Nearest-neighbor and
# next-nearest-neighbor bond sums below were verified by hand, by directly
# enumerating every periodic bond (not just re-deriving the implementation),
# so they pin down the *intended* value, not merely "whatever the code returns".
X = np.array(
    [
        [1, 1, -1],
        [1, -1, 1],
        [-1, 1, 1],
    ]
)


def test_nearest_neighbor_non_batched_value():
    field = Field(values=X, scale=1.0)
    # sum over all periodic bonds (vertical + horizontal) of x_i * x_j == -6
    assert NearestNeighbor().evaluate(field) == -6


def test_nearest_neighbor_batched_value():
    """Per-sample bond sum, one entry per batch element. -X gives the same
    value as X since the sign cancels in the product x_i * x_j."""
    field = Field(values=np.stack([X, -X]), scale=1.0, batched=True)
    np.testing.assert_array_equal(NearestNeighbor().evaluate(field), [-6, -6])


def test_next_nearest_neighbor_non_batched_value():
    field = Field(values=X, scale=1.0)
    # sum over all periodic diagonal bonds (down-right + down-left) == 6
    assert NextNearestNeighbor().evaluate(field) == 6


def test_next_nearest_neighbor_batched_value():
    field = Field(values=np.stack([X, -X]), scale=1.0, batched=True)
    np.testing.assert_array_equal(NextNearestNeighbor().evaluate(field), [6, 6])


def test_next_nearest_neighbor_requires_two_spatial_dims_non_batched():
    field = Field(values=np.array([1, -1, 1, 1, -1]), scale=1.0)
    with pytest.raises(ValueError):
        NextNearestNeighbor().evaluate(field)


def test_next_nearest_neighbor_requires_two_spatial_dims_batched():
    field = Field(values=np.array([[1, -1, 1], [1, 1, -1]]), scale=1.0, batched=True)
    with pytest.raises(ValueError):
        NextNearestNeighbor().evaluate(field)


def test_operator_basis_combines_operators_non_batched():
    field = Field(values=X, scale=1.0)
    basis = OperatorBasis([NearestNeighbor(), NextNearestNeighbor()])

    assert len(basis) == 2
    assert basis.names == ["K1", "K2"]
    np.testing.assert_array_equal(basis.evaluate(field), [-6, 6])


def test_operator_basis_combines_operators_batched():
    field = Field(values=np.stack([X, -X]), scale=1.0, batched=True)
    basis = OperatorBasis([NearestNeighbor(), NextNearestNeighbor()])

    np.testing.assert_array_equal(basis.evaluate(field), [[-6, -6], [6, 6]])
