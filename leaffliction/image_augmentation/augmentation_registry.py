"""Ordered registry of the six augmentations required by the subject."""

from leaffliction.image_augmentation.crop_augmentation import (
    CropAugmentation,
)
from leaffliction.image_augmentation.distortion_augmentation import (
    DistortionAugmentation,
)
from leaffliction.image_augmentation.flip_augmentation import (
    FlipAugmentation,
)
from leaffliction.image_augmentation.rotate_augmentation import (
    RotateAugmentation,
)
from leaffliction.image_augmentation.shear_augmentation import (
    ShearAugmentation,
)
from leaffliction.image_augmentation.skew_augmentation import (
    SkewAugmentation,
)
from leaffliction.utils.error_reporting import LeafflictionError

AUGMENTATION_CLASS_SEQUENCE = (
    FlipAugmentation,
    RotateAugmentation,
    SkewAugmentation,
    ShearAugmentation,
    CropAugmentation,
    DistortionAugmentation,
)


class UnknownAugmentationError(LeafflictionError):
    """Raised when an augmentation is requested by an unknown name."""


class AugmentationRegistry:
    """Expose the augmentations in one fixed, documented order.

    The order is the display order of the preview figure and the order
    in which the balancing stage cycles through the transformations, so
    it is defined once here rather than duplicated at every call site.
    """

    def build_default_augmentation_sequence(self):
        """Return one fresh instance of every augmentation, in order."""
        augmentation_sequence = []
        for augmentation_class in AUGMENTATION_CLASS_SEQUENCE:
            augmentation_sequence.append(augmentation_class())
        return augmentation_sequence

    def list_augmentation_names(self):
        """Return the ordered names of every registered augmentation."""
        augmentation_names = []
        for augmentation in self.build_default_augmentation_sequence():
            augmentation_names.append(augmentation.augmentation_name)
        return augmentation_names

    def get_augmentation_by_name(self, requested_augmentation_name):
        """Return the augmentation registered under that exact name."""
        for augmentation in self.build_default_augmentation_sequence():
            if augmentation.augmentation_name == requested_augmentation_name:
                return augmentation
        raise UnknownAugmentationError(
            "unknown augmentation: " + requested_augmentation_name
        )
