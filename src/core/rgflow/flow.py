from dataclasses import dataclass
from core.rgflow.couplings import Couplings


@dataclass
class RGFlow:

    points: list[Couplings]

    def add(self, couplings: Couplings):
        self.points.append(couplings)

    @property
    def scales(self):
        return [p.scale for p in self.points]

    def values(self, name: str):
        return [
            p[name]
            for p in self.points
        ]

    def __len__(self):
        return len(self.points)