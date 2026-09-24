"""Single definition of the image file formats handled by the project."""

SUPPORTED_IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})


def is_supported_image_extension(file_extension):
    """Return True when the extension names a handled image format.

    The comparison is case insensitive because the data set mixes the
    ".JPG" and ".jpg" spellings.
    """
    return file_extension.lower() in SUPPORTED_IMAGE_EXTENSIONS


def is_supported_image_file(candidate_path):
    """Return True when the path points to a handled image file."""
    if not candidate_path.is_file():
        return False
    return is_supported_image_extension(candidate_path.suffix)
