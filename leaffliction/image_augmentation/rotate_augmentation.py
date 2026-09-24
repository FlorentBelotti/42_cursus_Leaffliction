"""Rotate a leaf image around its center, keeping the frame size."""

from PIL import Image

from leaffliction.image_augmentation.augmentation_fill_color import (
    AUGMENTATION_FILL_COLOR,
)
from leaffliction.image_augmentation.base_augmentation import BaseAugmentation

AUGMENTATION_NAME = "Rotate"
DEFAULT_ROTATION_ANGLE_IN_DEGREES = 25
ROTATION_ANGLE_SEQUENCE_IN_DEGREES = (25, -25, 12, -12, 40, -40)


class RotateAugmentation(BaseAugmentation):
    """Rotate the image by a fixed angle, filling the freed corners."""

    def __init__(
        self,
        rotation_angle_in_degrees=DEFAULT_ROTATION_ANGLE_IN_DEGREES,
    ):
        """Store the rotation angle, counter clockwise and in degrees."""
        self.rotation_angle_in_degrees = rotation_angle_in_degrees

    @property
    def augmentation_name(self):
        """Return the file suffix and chart label of this augmentation."""
        return AUGMENTATION_NAME

    def apply_to_image(self, source_image):
        """Return the rotated image, kept at the original frame size.

        The frame size is preserved on purpose: every image of the
        augmented data set must keep the same shape, because part four
        feeds them to a network expecting a fixed input tensor.
        """
        return source_image.rotate(
            self.rotation_angle_in_degrees,
            resample=Image.Resampling.BICUBIC,
            expand=False,
            fillcolor=AUGMENTATION_FILL_COLOR,
        )

    def build_variant_with_index(self, variation_index):
        """Return a rotation walking the table of angles."""
        angle_index = variation_index % len(
            ROTATION_ANGLE_SEQUENCE_IN_DEGREES
        )
        return RotateAugmentation(
            ROTATION_ANGLE_SEQUENCE_IN_DEGREES[angle_index]
        )
