from abc import ABC, abstractmethod
from core.data.field import Field

import numpy as np


class Operator(ABC):
    name: str

    @abstractmethod
    def evaluate(self, field: Field) -> np.ndarray:
        pass


class OperatorBasis:

    def __init__(self, operators: list[Operator]):
        self.operators = list(operators)

    def evaluate(self, field: Field) -> np.ndarray:
        return np.array([
            op.evaluate(field)
            for op in self.operators
        ])

    @property
    def names(self):
        return [op.name for op in self.operators]

    def __len__(self):
        return len(self.operators)