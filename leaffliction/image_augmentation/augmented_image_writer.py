"""Name and write the augmented images on the disk.

The subject fixes the naming convention: the original file name,
followed by the name of the augmentation. When the balancing stage
needs several variants of the same pair, a variant number is appended,
so that no file ever silently overwrites another.
"""

VARIANT_SUFFIX_PREFIX = "_v"
JPEG_SAVE_QUALITY = 95


class AugmentedImageWriter:
    """Write augmented images under their conventional file name."""

    def __init__(self, output_directory_path=None):
        """Store where the images go, None meaning beside the source."""
        self.output_directory_path = output_directory_path

    def build_augmented_image_path(
        self,
        original_image_path,
        augmentation_name,
        variation_index=0,
    ):
        """Return the path the augmented image will be written to."""
        output_directory_path = self._resolve_output_directory_for(
            original_image_path
        )
        augmented_file_name = self._build_augmented_file_name(
            original_image_path,
            augmentation_name,
            variation_index,
        )
        return output_directory_path / augmented_file_name

    def write_augmented_image(
        self,
        augmented_image,
        original_image_path,
        augmentation_name,
        variation_index=0,
    ):
        """Write the image to the disk and return where it was written."""
        augmented_image_path = self.build_augmented_image_path(
            original_image_path,
            augmentation_name,
            variation_index,
        )
        augmented_image_path.parent.mkdir(parents=True, exist_ok=True)
        augmented_image.save(
            augmented_image_path,
            quality=JPEG_SAVE_QUALITY,
        )
        return augmented_image_path

    def _resolve_output_directory_for(self, original_image_path):
        """Return the directory the augmented image belongs to."""
        if self.output_directory_path is None:
            return original_image_path.parent
        return self.output_directory_path

    def _build_augmented_file_name(
        self,
        original_image_path,
        augmentation_name,
        variation_index,
    ):
        """Return the conventional file name of an augmented image."""
        variant_suffix = self._build_variant_suffix(variation_index)
        return (
            original_image_path.stem
            + "_"
            + augmentation_name
            + variant_suffix
            + original_image_path.suffix
        )

    def _build_variant_suffix(self, variation_index):
        """Return an empty suffix for the first variant, a number after."""
        if variation_index == 0:
            return ""
        return VARIANT_SUFFIX_PREFIX + str(variation_index + 1)
