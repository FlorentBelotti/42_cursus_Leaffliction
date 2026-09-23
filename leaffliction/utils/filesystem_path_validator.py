"""Turn raw command line paths into validated Path objects."""

from pathlib import Path

from leaffliction.utils.error_reporting import InvalidDirectoryError
from leaffliction.utils.error_reporting import InvalidImageFileError
from leaffliction.utils.supported_image_extensions import (
    is_supported_image_extension,
)


def validate_existing_directory(raw_directory_path):
    """Return the resolved path of an existing directory."""
    directory_path = Path(raw_directory_path).expanduser()
    if not directory_path.exists():
        raise InvalidDirectoryError(
            "directory does not exist: " + str(directory_path)
        )
    if not directory_path.is_dir():
        raise InvalidDirectoryError(
            "path is not a directory: " + str(directory_path)
        )
    return directory_path.resolve()


def validate_existing_image_file(raw_image_file_path):
    """Return the resolved path of an existing supported image file."""
    image_file_path = Path(raw_image_file_path).expanduser()
    if not image_file_path.exists():
        raise InvalidImageFileError(
            "image file does not exist: " + str(image_file_path)
        )
    if not image_file_path.is_file():
        raise InvalidImageFileError(
            "path is not a file: " + str(image_file_path)
        )
    if not is_supported_image_extension(image_file_path.suffix):
        raise InvalidImageFileError(
            "unsupported image extension: " + image_file_path.suffix
        )
    return image_file_path.resolve()


def resolve_writable_path(raw_path):
    """Return the resolved path of a file or directory to be created."""
    return Path(raw_path).expanduser().resolve()
