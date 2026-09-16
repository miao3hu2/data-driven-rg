from dataclasses import dataclass
import numpy as np

from .base import CoarseGrainer
from ..data.field import Field

@dataclass
class MajorityBlockSpin(CoarseGrainer):
    block_size: int = 2

    def transform(
        self,
        field: Field,
    ) -> Field:
        values = field.values
        if field.batched:
            batch_size, *spatial_dims = values.shape
            new_spatial_dims = [d // self.block_size for d in spatial_dims]
            blocked_shape = np.empty(2 * len(new_spatial_dims) + 1, dtype=int)
            blocked_shape[0] = batch_size
            blocked_shape[1::2] = new_spatial_dims
            blocked_shape[2::2] = self.block_size
            reshaped_values = values.reshape(blocked_shape)
            new_values = np.sign(np.sum(reshaped_values, axis=tuple(range(2, reshaped_values.ndim, 2)))) # A majority rule for block spins
            new_values[new_values == 0] = 1  # Handle the case where the sum is zero
            
        else:
            spatial_dims = values.shape
            new_spatial_dims = [d // self.block_size for d in spatial_dims]
            blocked_shape = np.empty(2 * len(new_spatial_dims), dtype=int)
            blocked_shape[0::2] = new_spatial_dims
            blocked_shape[1::2] = self.block_size
            reshaped_values = values.reshape(blocked_shape)
            new_values = np.sign(np.sum(reshaped_values, axis=tuple(range(1, reshaped_values.ndim, 2)))) # A majority rule for block spins
            new_values[new_values == 0] = 1  # Handle the case where the sum is zero

        new_scale = tuple(s * self.block_size for s in field.scale)
        return Field(values=new_values, scale=new_scale, batched=field.batched, metadata=field.metadata)