from abc import ABC, abstractmethod
from core.data.base import Field

import numpy as np


class Operator(ABC):
    name: str

    @abstractmethod
    def evaluate(self, field: Field) -> np.ndarray:
        pass


class OperatorBasis:

    def __init__(self, operators: list[Operator] | None = None):
        self.operators = list(operators) if operators is not None else []

    def evaluate(self, field: Field) -> np.ndarray:
        return np.array([
            op.evaluate(field)
            for op in self.operators
        ])

    def add_operator(self, operator: Operator):
        self.operators.append(operator)

    @property
    def names(self):
        return [op.name for op in self.operators]

    def __len__(self):
        return len(self.operators)