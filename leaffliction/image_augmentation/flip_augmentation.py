"""Mirror a leaf image along one of its two axes."""

from PIL import ImageOps

from leaffliction.image_augmentation.base_augmentation import BaseAugmentation

AUGMENTATION_NAME = "Flip"
HORIZONTAL_FLIP_DIRECTION = "horizontal"
VERTICAL_FLIP_DIRECTION = "vertical"
FLIP_DIRECTION_SEQUENCE = (
    HORIZONTAL_FLIP_DIRECTION,
    VERTICAL_FLIP_DIRECTION,
)


class FlipAugmentation(BaseAugmentation):
    """Mirror the image horizontally or vertically."""

    def __init__(self, flip_direction=HORIZONTAL_FLIP_DIRECTION):
        """Store the axis the image will be mirrored along."""
        self.flip_direction = flip_direction

    @property
    def augmentation_name(self):
        """Return the file suffix and chart label of this augmentation."""
        return AUGMENTATION_NAME

    def apply_to_image(self, source_image):
        """Return the mirrored image, without any resampling loss."""
        if self.flip_direction == HORIZONTAL_FLIP_DIRECTION:
            return ImageOps.mirror(source_image)
        else:
            return ImageOps.flip(source_image)

    def build_variant_with_index(self, variation_index):
        """Return a flip alternating between the two axes."""
        direction_index = variation_index % len(FLIP_DIRECTION_SEQUENCE)
        return FlipAugmentation(FLIP_DIRECTION_SEQUENCE[direction_index])
