"""Distort a leaf image by displacing its pixels along a sine wave."""

import numpy
from PIL import Image

from leaffliction.image_augmentation.base_augmentation import BaseAugmentation

AUGMENTATION_NAME = "Distortion"
DEFAULT_WAVE_AMPLITUDE_IN_PIXELS = 12.0
DEFAULT_WAVE_LENGTH_IN_PIXELS = 90.0

WAVE_PARAMETER_SEQUENCE = (
    (12.0, 90.0),
    (18.0, 130.0),
    (8.0, 60.0),
    (15.0, 200.0),
)


class DistortionAugmentation(BaseAugmentation):
    """Ripple the image, as if the leaf were seen through water.

    The transformation is expressed as an index remapping: for every
    output pixel a displaced source coordinate is computed, then the
    whole picture is gathered in one indexing operation. Coordinates
    falling outside the frame are clamped, which replicates the border
    pixels instead of leaving a hole.
    """

    def __init__(
        self,
        wave_amplitude_in_pixels=DEFAULT_WAVE_AMPLITUDE_IN_PIXELS,
        wave_length_in_pixels=DEFAULT_WAVE_LENGTH_IN_PIXELS,
    ):
        """Store how far and how often the pixels are displaced."""
        self.wave_amplitude_in_pixels = wave_amplitude_in_pixels
        self.wave_length_in_pixels = wave_length_in_pixels

    @property
    def augmentation_name(self):
        """Return the file suffix and chart label of this augmentation."""
        return AUGMENTATION_NAME

    def apply_to_image(self, source_image):
        """Return the rippled image, at the original frame size."""
        source_pixel_array = numpy.asarray(source_image)
        image_height = source_pixel_array.shape[0]
        image_width = source_pixel_array.shape[1]
        displaced_rows, displaced_columns = self._build_displaced_grids(
            image_height,
            image_width,
        )
        distorted_pixel_array = source_pixel_array[
            displaced_rows,
            displaced_columns,
        ]
        return Image.fromarray(distorted_pixel_array)

    def _build_displaced_grids(self, image_height, image_width):
        """Return the source row and column index of every output pixel."""
        row_grid, column_grid = numpy.meshgrid(
            numpy.arange(image_height),
            numpy.arange(image_width),
            indexing="ij",
        )
        row_displacement = self._compute_sine_displacement(column_grid)
        column_displacement = self._compute_sine_displacement(row_grid)
        displaced_rows = self._clamp_grid_to_frame(
            row_grid + row_displacement,
            image_height,
        )
        displaced_columns = self._clamp_grid_to_frame(
            column_grid + column_displacement,
            image_width,
        )
        return displaced_rows, displaced_columns

    def _compute_sine_displacement(self, coordinate_grid):
        """Return the displacement driven by a sine of the coordinate."""
        wave_phase = (
            2.0 * numpy.pi * coordinate_grid / self.wave_length_in_pixels
        )
        return self.wave_amplitude_in_pixels * numpy.sin(wave_phase)

    def _clamp_grid_to_frame(self, coordinate_grid, frame_length):
        """Round the coordinates and keep them inside the frame."""
        rounded_grid = numpy.round(coordinate_grid).astype(numpy.int64)
        return numpy.clip(rounded_grid, 0, frame_length - 1)

    def build_variant_with_index(self, variation_index):
        """Return a distortion walking the table of wave parameters."""
        parameter_index = variation_index % len(WAVE_PARAMETER_SEQUENCE)
        wave_parameters = WAVE_PARAMETER_SEQUENCE[parameter_index]
        return DistortionAugmentation(
            wave_amplitude_in_pixels=wave_parameters[0],
            wave_length_in_pixels=wave_parameters[1],
        )
