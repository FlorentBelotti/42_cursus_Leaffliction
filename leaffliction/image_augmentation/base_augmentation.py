"""Common contract shared by every augmentation transformation."""

from abc import ABC
from abc import abstractmethod


class BaseAugmentation(ABC):
    """Turn one leaf image into one augmented variant of that image.

    Every concrete augmentation is deterministic: the same source image
    and the same variation index always produce the same output. That
    property keeps the augmented data set reproducible, which matters
    because its sha1 signature is turned in alongside the code.

    Parameter diversity is obtained through build_variant_with_index
    rather than through a random generator: the balancing stage asks for
    variant zero, then variant one, and so on, walking a fixed table of
    parameters instead of drawing them.
    """

    @property
    @abstractmethod
    def augmentation_name(self):
        """Return the name used as file suffix and as chart label."""

    @abstractmethod
    def apply_to_image(self, source_image):
        """Return a new augmented image built from the source image."""

    @abstractmethod
    def build_variant_with_index(self, variation_index):
        """Return the same augmentation with shifted parameters."""
