from core.rgflow.flow import RGFlow
from core.rgflow.couplings import Couplings


def test_add_couplings():

    flow = RGFlow(points=[])

    g1 = Couplings(
        {"K": 0.44},
        scale=1,
    )

    g2 = Couplings(
        {"K": 0.42}, 
        scale=2
    )

    flow.add(g1)

    flow.add(g2)

    assert len(flow) == 2
    assert flow.scales == [1, 2]
    assert flow.values("K") == [0.44, 0.42]