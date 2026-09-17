from core.data.ising import Ising2D
import numpy as np

def test_ising2d_sampling():
    ising_model = Ising2D(L=4, beta=0.44, seed=42)
    data = ising_model.sample(n_samples=5, burn_in=100, thinning=10)

    assert data.values.shape == (5, 4, 4)
    assert data.scale == (1.0, 1.0)
    assert np.all(np.isin(data.values, [-1, 1]))
