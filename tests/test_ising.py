import numpy as np
import pytest

from core.data.ising import Ising2D


@pytest.fixture
def sample_data():
    model = Ising2D(L=4, beta=0.44, seed=42)
    return model.sample(n_samples=5, burn_in=100, thinning=10)


def test_sample_shape(sample_data):
    assert sample_data.values.shape == (5, 4, 4)


def test_sample_scale(sample_data):
    assert sample_data.scale == (1.0, 1.0)


def test_sample_values_are_plus_minus_one(sample_data):
    assert np.all(np.isin(sample_data.values, [-1, 1]))


def test_sample_metadata():
    model = Ising2D(L=4, beta=0.37, J=1.5, seed=1)
    data = model.sample(n_samples=2, burn_in=10, thinning=2)
    assert data.metadata == {"system": "2d_ising", "beta": 0.37, "J": 1.5}


def test_sampling_is_deterministic_given_seed():
    data1 = Ising2D(L=4, beta=0.44, seed=123).sample(n_samples=3, burn_in=50, thinning=5)
    data2 = Ising2D(L=4, beta=0.44, seed=123).sample(n_samples=3, burn_in=50, thinning=5)
    np.testing.assert_array_equal(data1.values, data2.values)


def test_energy_change_matches_manual_calculation():
    spins = np.array(
        [
            [1, 1, -1, 1],
            [-1, 1, 1, -1],
            [1, -1, 1, 1],
            [1, 1, -1, -1],
        ]
    )
    model = Ising2D(L=4, beta=0.44, J=2.0, seed=0)

    # site (0, 0): s=1, periodic neighbors are spins[1,0]=-1, spins[3,0]=1,
    # spins[0,1]=1, spins[0,3]=1 -> nn sum = 2 -> dE = 2*J*s*nn = 2*2*1*2 = 8
    assert model.energy_change(spins, 0, 0) == 8.0

    # site (1, 0): s=-1, periodic neighbors are spins[2,0]=1, spins[0,0]=1,
    # spins[1,1]=1, spins[1,3]=-1 -> nn sum = 2 -> dE = 2*2*(-1)*2 = -8
    assert model.energy_change(spins, 1, 0) == -8.0
