from abc import ABC, abstractmethod

from ..data.field import Field


class CoarseGrainer(ABC):

    @abstractmethod
    def transform(
        self,
        field: Field,
    ) -> Field:
        pass

    def __call__(self, field: Field) -> Field:
        return self.transform(field)