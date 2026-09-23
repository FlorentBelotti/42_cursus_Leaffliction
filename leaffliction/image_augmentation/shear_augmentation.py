"""Shear a leaf image, sliding its rows sideways."""

from PIL import Image

from leaffliction.image_augmentation.augmentation_fill_color import (
    AUGMENTATION_FILL_COLOR,
)
from leaffliction.image_augmentation.base_augmentation import BaseAugmentation

AUGMENTATION_NAME = "Shear"
DEFAULT_SHEAR_FACTOR = 0.32
SHEAR_FACTOR_SEQUENCE = (0.32, -0.32, 0.18, -0.18)


class ShearAugmentation(BaseAugmentation):
    """Slide every row of the image sideways, proportionally to its y.

    This is an affine transformation: the frame becomes a parallelogram
    and parallel lines stay parallel, unlike the Skew augmentation.
    """

    def __init__(self, shear_factor=DEFAULT_SHEAR_FACTOR):
        """Store how far a row slides per pixel of vertical offset."""
        self.shear_factor = shear_factor

    @property
    def augmentation_name(self):
        """Return the file suffix and chart label of this augmentation."""
        return AUGMENTATION_NAME

    def apply_to_image(self, source_image):
        """Return the sheared image, recentered inside the same frame."""
        image_width, image_height = source_image.size
        affine_coefficients = self._build_affine_coefficients(image_height)
        return source_image.transform(
            (image_width, image_height),
            Image.Transform.AFFINE,
            affine_coefficients,
            resample=Image.Resampling.BICUBIC,
            fillcolor=AUGMENTATION_FILL_COLOR,
        )

    def _build_affine_coefficients(self, image_height):
        """Return the six affine coefficients of a horizontal shear.

        Pillow reads the source pixel at
        (a * x + b * y + c, d * x + e * y + f) for every output pixel.
        The constant term c recenters the parallelogram, so that the
        leaf does not drift out of the frame on one side.
        """
        horizontal_recentering_offset = (
            -self.shear_factor * image_height / 2.0
        )
        return (
            1.0,
            self.shear_factor,
            horizontal_recentering_offset,
            0.0,
            1.0,
            0.0,
        )

    def build_variant_with_index(self, variation_index):
        """Return a shear walking the table of factors."""
        factor_index = variation_index % len(SHEAR_FACTOR_SEQUENCE)
        return ShearAugmentation(SHEAR_FACTOR_SEQUENCE[factor_index])
