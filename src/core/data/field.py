from dataclasses import dataclass
import numpy as np
from typing import Sequence

class Grid:
    def __init__(self, shape, spacing):
        self.shape = shape
        self.spacing = spacing
        self.coordinates = self._generate_coordinates()

    def _generate_coordinates(self):
        return np.meshgrid(*[np.arange(0, s) * sp for s, sp in zip(self.shape, self.spacing)], indexing='ij')

@dataclass
class Field:
    values: np.ndarray
    scale: float | Sequence[float]
    batched: bool = False
    metadata: dict | None = None

    def __post_init__(self):
        self.values = np.asarray(self.values)
        n_spatial_dims = self.values.ndim - (1 if self.batched else 0)

        if np.ndim(self.scale) == 0:
            self.scale = (float(self.scale),) * n_spatial_dims
        else:
            self.scale = tuple(float(s) for s in self.scale)
            if len(self.scale) == 1 and n_spatial_dims != 1:
                self.scale = self.scale * n_spatial_dims
                
        if len(self.scale) != n_spatial_dims:
            raise ValueError("Scale length must match the number of spatial dimensions in values.")

        if self.metadata is None:
            self.metadata = {}

    @property
    def shape(self):
        return self.values.shape
