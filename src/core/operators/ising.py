import numpy as np
from core.data.base import Field
from core.operators.base import Operator
from itertools import combinations


class NearestNeighbor(Operator):

    name = "K1"

    def evaluate(self, field: Field) -> np.ndarray:

        x = field.values

        if field.batched:
            _, *spatial_dims = field.shape

            num_spatial_dims = len(spatial_dims)

            rolled_x = np.asarray([np.roll(x, -1, axis=i + 1) for i in range(num_spatial_dims)])

            return np.sum(x * rolled_x, axis=tuple(i for i in range(num_spatial_dims+2) if i != 1))

        else:
            *spatial_dims, = field.shape
            num_spatial_dims = len(spatial_dims)
            rolled_x = np.asarray([np.roll(x, -1, axis=i) for i in range(num_spatial_dims)])

            return np.asarray(np.sum(x * rolled_x))


class NextNearestNeighbor(Operator):

    name = "K2"

    def evaluate(self, field: Field) -> np.ndarray:

        x = field.values

        if field.batched:
            _, *spatial_dims = field.shape
            num_spatial_dims = len(spatial_dims)
            if num_spatial_dims < 2:
                raise ValueError("NextNearestNeighbor operator requires at least 2 spatial dimensions.")

            else:
                axes = list(combinations(range(1, num_spatial_dims+1), 2))

                diagonal_1 = np.asarray([np.roll(x, (-1, -1), axis = axis) for axis in axes])
                diagonal_2 = np.asarray([np.roll(x, (-1, 1), axis = axis) for axis in axes])

                return np.sum(x * diagonal_1 + x * diagonal_2, axis = tuple(i for i in range(num_spatial_dims+2) if i != 1))

        else:
            *spatial_dims, = field.shape
            num_spatial_dims = len(spatial_dims)
            if num_spatial_dims < 2:
                raise ValueError("NextNearestNeighbor operator requires at least 2 spatial dimensions.")

            else:
                axes = list(combinations(range(num_spatial_dims), 2))

                diagonal_1 = np.asarray([np.roll(x, (-1, -1), axis = axis) for axis in axes])
                diagonal_2 = np.asarray([np.roll(x, (-1, 1), axis = axis) for axis in axes])

                return np.asarray(np.sum(x * diagonal_1 + x * diagonal_2))
