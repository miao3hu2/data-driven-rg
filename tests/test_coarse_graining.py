import numpy as np

from core.data.field import Field
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


def test_majority_block_spin_non_batched_values():
    # Blocks (block_size=2), by (row, col) index:
    #   (0,0): [[1,1],[1,-1]]  sum= 2 -> majority  1
    #   (0,1): [[1,-1],[-1,-1]] sum=-2 -> majority -1
    #   (1,0): [[1,-1],[-1,1]]  sum= 0 -> tie, forced to  1
    #   (1,1): [[-1,-1],[-1,-1]] sum=-4 -> majority -1
    values = np.array(
        [
            [1, 1, 1, -1],
            [1, -1, -1, -1],
            [1, -1, -1, -1],
            [-1, 1, -1, -1],
        ]
    )
    field = Field(values=values, scale=1.0)
    cg = MajorityBlockSpin(block_size=2)
    blocked = cg.transform(field)

    expected = np.array(
        [
            [1, -1],
            [1, -1],
        ]
    )
    np.testing.assert_array_equal(blocked.values, expected)
    assert blocked.scale == (2.0, 2.0)


def test_majority_block_spin_batched_values():
    base = np.array(
        [
            [1, 1, 1, -1],
            [1, -1, -1, -1],
            [1, -1, -1, -1],
            [-1, 1, -1, -1],
        ]
    )
    values = np.stack([base, -base])
    field = Field(values=values, scale=1.0, batched=True)
    cg = MajorityBlockSpin(block_size=2)
    blocked = cg.transform(field)

    expected = np.array(
        [
            [[1, -1], [1, -1]],
            # tie block (0 sum) still forced to 1 rather than mirroring the flip
            [[-1, 1], [1, 1]],
        ]
    )
    np.testing.assert_array_equal(blocked.values, expected)
    assert blocked.scale == (2.0, 2.0)