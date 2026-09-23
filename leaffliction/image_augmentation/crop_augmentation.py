"""Crop a region of a leaf image and blow it back up to full frame."""

from PIL import Image

from leaffliction.image_augmentation.base_augmentation import BaseAugmentation

AUGMENTATION_NAME = "Crop"
DEFAULT_CROP_RATIO = 0.72
DEFAULT_HORIZONTAL_ANCHOR_RATIO = 0.5
DEFAULT_VERTICAL_ANCHOR_RATIO = 0.5

CROP_PARAMETER_SEQUENCE = (
    (0.72, 0.5, 0.5),
    (0.62, 0.3, 0.3),
    (0.62, 0.7, 0.7),
    (0.80, 0.3, 0.7),
    (0.80, 0.7, 0.3),
)


class CropAugmentation(BaseAugmentation):
    """Zoom into a sub region of the image, keeping the frame size."""

    def __init__(
        self,
        crop_ratio=DEFAULT_CROP_RATIO,
        horizontal_anchor_ratio=DEFAULT_HORIZONTAL_ANCHOR_RATIO,
        vertical_anchor_ratio=DEFAULT_VERTICAL_ANCHOR_RATIO,
    ):
        """Store the size of the kept region and where it is taken."""
        self.crop_ratio = crop_ratio
        self.horizontal_anchor_ratio = horizontal_anchor_ratio
        self.vertical_anchor_ratio = vertical_anchor_ratio

    @property
    def augmentation_name(self):
        """Return the file suffix and chart label of this augmentation."""
        return AUGMENTATION_NAME

    def apply_to_image(self, source_image):
        """Return the cropped region resized back to the frame size."""
        image_width, image_height = source_image.size
        crop_box = self._build_crop_box(image_width, image_height)
        cropped_image = source_image.crop(crop_box)
        return cropped_image.resize(
            (image_width, image_height),
            resample=Image.Resampling.BICUBIC,
        )

    def _build_crop_box(self, image_width, image_height):
        """Return the left, top, right and bottom of the kept region."""
        crop_width = int(image_width * self.crop_ratio)
        crop_height = int(image_height * self.crop_ratio)
        left_offset = self._compute_anchored_offset(
            image_width,
            crop_width,
            self.horizontal_anchor_ratio,
        )
        top_offset = self._compute_anchored_offset(
            image_height,
            crop_height,
            self.vertical_anchor_ratio,
        )
        return (
            left_offset,
            top_offset,
            left_offset + crop_width,
            top_offset + crop_height,
        )

    def _compute_anchored_offset(
        self,
        frame_length,
        crop_length,
        anchor_ratio,
    ):
        """Return the offset of the region along one axis of the frame."""
        available_travel = frame_length - crop_length
        return int(available_travel * anchor_ratio)

    def build_variant_with_index(self, variation_index):
        """Return a crop walking the table of ratios and anchors."""
        parameter_index = variation_index % len(CROP_PARAMETER_SEQUENCE)
        crop_parameters = CROP_PARAMETER_SEQUENCE[parameter_index]
        return CropAugmentation(
            crop_ratio=crop_parameters[0],
            horizontal_anchor_ratio=crop_parameters[1],
            vertical_anchor_ratio=crop_parameters[2],
        )
