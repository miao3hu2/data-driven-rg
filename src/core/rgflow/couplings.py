from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass
class Couplings:

    values: dict[str, float]
    scale: float | Sequence[float]

    def __getitem__(self, name: str) -> float:
        return self.values[name]

    def get(self, name: str, default=None):
        return self.values.get(name, default)

    def as_array(self) -> np.ndarray:
        return np.array(
            list(self.values.values()),
            dtype=float,
        )