import numpy as np
import pytest

from core.data.field import Field, Grid


def test_field_scalar_scale_broadcasts_to_all_spatial_dims():
    field = Field(values=np.zeros((3, 4)), scale=1.5)
    assert field.scale == (1.5, 1.5)


def test_field_single_element_scale_broadcasts_to_all_spatial_dims():
    field = Field(values=np.zeros((3, 4)), scale=[2.0])
    assert field.scale == (2.0, 2.0)


def test_field_sequence_scale_kept_as_given():
    field = Field(values=np.zeros((3, 4)), scale=(1.0, 2.0))
    assert field.scale == (1.0, 2.0)


def test_field_mismatched_scale_length_raises():
    with pytest.raises(ValueError):
        Field(values=np.zeros((3, 4)), scale=(1.0, 2.0, 3.0))


def test_field_batched_scale_excludes_batch_dim():
    field = Field(values=np.zeros((5, 3, 4)), scale=1.0, batched=True)
    assert field.scale == (1.0, 1.0)


def test_field_default_metadata_is_empty_dict():
    field = Field(values=np.zeros((2, 2)), scale=1.0)
    assert field.metadata == {}


def test_field_shape_property_matches_values_shape():
    values = np.zeros((3, 4))
    field = Field(values=values, scale=1.0)
    assert field.shape == (3, 4)


def test_field_values_coerced_to_ndarray():
    field = Field(values=[[1, 2], [3, 4]], scale=1.0)
    assert isinstance(field.values, np.ndarray)
    np.testing.assert_array_equal(field.values, [[1, 2], [3, 4]])


def test_grid_coordinates_shape_and_values():
    grid = Grid(shape=(3, 2), spacing=(1.0, 2.0))
    x, y = grid.coordinates

    assert x.shape == (3, 2)
    assert y.shape == (3, 2)

    np.testing.assert_array_equal(x[:, 0], [0.0, 1.0, 2.0])
    np.testing.assert_array_equal(y[0, :], [0.0, 2.0])
