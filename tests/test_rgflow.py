import pytest

from core.rgflow.flow import RGFlow
from core.rgflow.couplings import Couplings


def _couplings(K, scale):
    return Couplings({"K": K}, scale=scale)


def test_flow_len_and_scales():
    flow = RGFlow(points=[_couplings(0.44, 1), _couplings(0.42, 2)])
    assert len(flow) == 2
    assert flow.scales == [1, 2]


def test_flow_values_returns_per_point_coupling():
    flow = RGFlow(points=[_couplings(0.44, 1), _couplings(0.42, 2)])
    assert flow.values("K") == [0.44, 0.42]


def test_add_appends_a_point():
    flow = RGFlow(points=[_couplings(0.44, 1)])
    flow.add(_couplings(0.42, 2))
    assert len(flow) == 2
    assert flow.scales == [1, 2]


def test_empty_flow_has_no_points():
    flow = RGFlow(points=[])
    assert len(flow) == 0
    assert flow.scales == []


def test_values_raises_keyerror_for_missing_coupling():
    flow = RGFlow(points=[_couplings(0.44, 1)])
    with pytest.raises(KeyError):
        flow.values("h")
