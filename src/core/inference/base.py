from abc import ABC, abstractmethod
from core.data.base import Field
from core.operators.base import OperatorBasis
from core.rgflow.couplings import CouplingsForOperators


class InferenceMethod(ABC):

    @abstractmethod
    def fit(self, field: Field, basis: OperatorBasis) -> CouplingsForOperators:
        pass

def infer_couplings(field: Field, basis: OperatorBasis, method: InferenceMethod):
    return method.fit(field, basis)