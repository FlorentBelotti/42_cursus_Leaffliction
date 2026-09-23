"""Skew a leaf image with a perspective, as seen from an angle."""

from PIL import Image

from leaffliction.image_augmentation.augmentation_fill_color import (
    AUGMENTATION_FILL_COLOR,
)
from leaffliction.image_augmentation.base_augmentation import BaseAugmentation
from leaffliction.utils.perspective_coefficient_solver import (
    find_perspective_transform_coefficients,
)

AUGMENTATION_NAME = "Skew"
DEFAULT_SKEW_INTENSITY_RATIO = 0.22
SKEW_INTENSITY_RATIO_SEQUENCE = (0.22, 0.30, 0.14, 0.26)


class SkewAugmentation(BaseAugmentation):
    """Narrow the top edge of the image to fake a viewing angle.

    This is a true perspective transformation: parallel lines of the
    source stop being parallel in the result. It is what visually
    distinguishes Skew from Shear, which stays affine.
    """

    def __init__(self, skew_intensity_ratio=DEFAULT_SKEW_INTENSITY_RATIO):
        """Store the horizontal inset of the top edge, as a ratio."""
        self.skew_intensity_ratio = skew_intensity_ratio

    @property
    def augmentation_name(self):
        """Return the file suffix and chart label of this augmentation."""
        return AUGMENTATION_NAME

    def apply_to_image(self, source_image):
        """Return the image mapped onto a trapezoid of the same frame."""
        image_width, image_height = source_image.size
        source_corner_points = self._build_source_corner_points(
            image_width,
            image_height,
        )
        destination_corner_points = self._build_trapezoid_corner_points(
            image_width,
            image_height,
        )
        perspective_coefficients = find_perspective_transform_coefficients(
            source_corner_points,
            destination_corner_points,
        )
        return source_image.transform(
            (image_width, image_height),
            Image.Transform.PERSPECTIVE,
            perspective_coefficients,
            resample=Image.Resampling.BICUBIC,
            fillcolor=AUGMENTATION_FILL_COLOR,
        )

    def _build_source_corner_points(self, image_width, image_height):
        """Return the four corners of the source frame, clockwise."""
        return (
            (0, 0),
            (image_width, 0),
            (image_width, image_height),
            (0, image_height),
        )

    def _build_trapezoid_corner_points(self, image_width, image_height):
        """Return the destination trapezoid, with a narrowed top edge."""
        horizontal_inset = int(image_width * self.skew_intensity_ratio)
        return (
            (horizontal_inset, 0),
            (image_width - horizontal_inset, 0),
            (image_width, image_height),
            (0, image_height),
        )

    def build_variant_with_index(self, variation_index):
        """Return a skew walking the table of intensities."""
        intensity_index = variation_index % len(
            SKEW_INTENSITY_RATIO_SEQUENCE
        )
        return SkewAugmentation(
            SKEW_INTENSITY_RATIO_SEQUENCE[intensity_index]
        )
