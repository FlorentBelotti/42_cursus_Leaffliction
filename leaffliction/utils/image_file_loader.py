"""Single entry point for reading an image file from the disk."""

from PIL import Image
from PIL import UnidentifiedImageError

from leaffliction.utils.error_reporting import InvalidImageFileError


def load_image_as_rgb(image_file_path):
    """Open an image file and return it as an RGB Pillow image.

    Converting to RGB up front removes every palette, grayscale and
    alpha channel special case from the augmentation code downstream.
    """
    try:
        opened_image = Image.open(image_file_path)
        return opened_image.convert("RGB")
    except (OSError, UnidentifiedImageError) as opening_error:
        raise InvalidImageFileError(
            "cannot read image file: " + str(image_file_path)
        ) from opening_error
