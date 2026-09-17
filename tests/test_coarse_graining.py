from core.data.ising import Ising2D
from core.coarse_graining.block import MajorityBlockSpin

def test_majority_block_spin():

    ising_model = Ising2D(L=8, beta=0.44, seed=42)
    data = ising_model.sample(n_samples=10, burn_in=100, thinning=10)
    cg = MajorityBlockSpin(block_size=2)
    blocked_data1 = cg.transform(data)
    blocked_data2 = cg.transform(blocked_data1)

    assert blocked_data1.values.shape == (10, 4, 4)
    assert blocked_data1.scale == (2.0, 2.0)
    assert blocked_data2.values.shape == (10, 2, 2)
    assert blocked_data2.scale == (4.0, 4.0)