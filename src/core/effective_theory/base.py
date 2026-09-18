from dataclasses import dataclass
from core.rgflow.couplings import CouplingsForOperators
from core.operators.base import OperatorBasis

@dataclass
class EffectiveTheory:

    basis: OperatorBasis
    couplings: CouplingsForOperators

    def __post_init__(self):
        if self.basis.names != self.couplings.values.keys:
            raise ValueError(f"There is a mismatch between the operators and couplings! Got operators [{self.basis.names}] and couplings [{self.couplings.values}]")

    def describe(self):
        terms = []

        for name, value in self.couplings.values.items():
            terms.append(
                f"{value:+.6f} * {name}"
            )

        return "H_eff = " + " ".join(terms)