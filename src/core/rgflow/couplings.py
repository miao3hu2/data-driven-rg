from typing import Sequence
import numpy as np
from core.operators.base import OperatorBasis


class CouplingsForOperators:

    def __init__(self, basis: OperatorBasis, values: np.ndarray, scale: float | Sequence[float]):

        values = np.atleast_1d(values)

        if len(values) != len(basis):
            raise ValueError(f"expected {len(basis)} values, got {len(values)}")
        if len(set(basis.names)) != len(basis):
            raise ValueError("basis contains duplicate operator names")
        self.values = dict(zip(basis.names, values))
        self.scale = scale

    def __getitem__(self, name: str) -> float:
        return self.values[name]

    def get(self, name: str, default=None):
        return self.values.get(name, default)

    def as_array(self) -> np.ndarray:
        return np.array(
            list(self.values.values()),
            dtype=float,
        )