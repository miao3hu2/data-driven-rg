import numpy as np

from core.rgflow.couplings import Couplings


def test_getitem_returns_value():
    c = Couplings(values={"K": 0.44, "h": 0.1}, scale=1.0)
    assert c["K"] == 0.44
    assert c["h"] == 0.1


def test_get_returns_default_when_missing():
    c = Couplings(values={"K": 0.44}, scale=1.0)
    assert c.get("missing") is None
    assert c.get("missing", 0.0) == 0.0
    assert c.get("K", 0.0) == 0.44


def test_as_array_preserves_insertion_order_and_values():
    c = Couplings(values={"K": 0.44, "h": 0.1, "g": -1.5}, scale=1.0)
    np.testing.assert_array_equal(c.as_array(), [0.44, 0.1, -1.5])
    assert c.as_array().dtype == np.float64
