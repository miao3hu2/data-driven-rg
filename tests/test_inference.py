"""Correctness checks for InferK1 (src/core/inference/ising.py).

Two things are pinned down separately:
  - _neighbor_field, against a hand-computed periodic nearest-neighbor sum
    (same 3x3 lattice and derivation style as test_operators.py), and
  - fit(), against configurations where the sign/extremum of the maximum
    (pseudo-)likelihood estimate is known analytically, plus the physical
    invariances (global spin flip, translation) the estimator must respect.
"""

import numpy as np
import pytest

from core.data.base import Field
from core.operators.base import OperatorBasis
from core.operators.ising import NearestNeighbor
from core.inference.ising import InferK1


def _k1_basis():
    return OperatorBasis([NearestNeighbor()])

# 3x3 periodic spin configuration reused from test_operators.py.
X = np.array(
    [
        [1, 1, -1],
        [1, -1, 1],
        [-1, 1, 1],
    ],
    dtype=float,
)

# Hand-computed periodic 4-neighbor sum h[i,j] = x[i-1,j]+x[i+1,j]+x[i,j-1]+x[i,j+1]
# (mod 3), derived by directly enumerating each site's four periodic neighbors.
EXPECTED_NEIGHBOR_SUM = np.array(
    [
        [0, 0, 4],
        [0, 4, 0],
        [4, 0, 0],
    ],
    dtype=float,
)


def test_neighbor_field_matches_hand_verified_sum_batched():
    field = Field(values=X[None, :, :], scale=1.0, batched=True)
    result = InferK1()._neighbor_field(field)
    np.testing.assert_array_equal(result[0], EXPECTED_NEIGHBOR_SUM)


def test_neighbor_field_matches_hand_verified_sum_non_batched():
    field = Field(values=X, scale=1.0)
    result = InferK1()._neighbor_field(field)
    np.testing.assert_array_equal(result, EXPECTED_NEIGHBOR_SUM)


def test_fit_recovers_upper_bound_for_saturated_ferromagnetic_field():
    # Every spin is +1 and every neighbor sum is +4, so x*h > 0 everywhere:
    # the (pseudo-)likelihood increases monotonically with K1, so the bounded
    # optimizer (bounds=(-2, 2)) must land at the upper bound.
    x = np.ones((1, 4, 4))
    field = Field(values=x, scale=1.0, batched=True)

    couplings = InferK1().fit(field, _k1_basis())

    assert couplings["K1"] == pytest.approx(2.0, abs=1e-2)


def test_fit_recovers_lower_bound_for_checkerboard_field():
    # A checkerboard pattern makes every spin anti-aligned with all four of
    # its neighbors (x*h == -4 everywhere), so the likelihood increases
    # monotonically as K1 decreases, and the optimizer must land at -2.
    L = 4
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    checkerboard = np.where((i + j) % 2 == 0, 1.0, -1.0)
    field = Field(values=checkerboard[None, :, :], scale=1.0, batched=True)

    couplings = InferK1().fit(field, _k1_basis())

    assert couplings["K1"] == pytest.approx(-2.0, abs=1e-2)


def test_fit_invariant_under_global_spin_flip():
    # x*h is unchanged when every spin (and hence every neighbor sum) flips
    # sign, so the estimate must be identical for a configuration and its
    # global spin-flip.
    x = np.random.default_rng(0).choice([-1.0, 1.0], size=(1, 6, 6))
    basis = _k1_basis()

    k1 = InferK1().fit(Field(values=x, scale=1.0, batched=True), basis)["K1"]
    k1_flipped = InferK1().fit(Field(values=-x, scale=1.0, batched=True), basis)["K1"]

    assert k1 == pytest.approx(k1_flipped)


def test_fit_invariant_under_translation():
    # On a periodic lattice, rolling the configuration relabels sites but not
    # which pairs are bonded, so the inferred coupling must be unchanged.
    x = np.random.default_rng(1).choice([-1.0, 1.0], size=(1, 6, 6))
    shifted = np.roll(x, shift=(0, 2, 3), axis=(0, 1, 2))
    basis = _k1_basis()

    k1 = InferK1().fit(Field(values=x, scale=1.0, batched=True), basis)["K1"]
    k1_shifted = InferK1().fit(Field(values=shifted, scale=1.0, batched=True), basis)["K1"]

    assert k1 == pytest.approx(k1_shifted)
