"""The two things the Augmentation entrypoint can be asked to do.

A single executable covers both so that the evaluator finds everything
from Augmentation.py: augmenting one image for display, and rebuilding
a balanced copy of a whole data set.
"""

from leaffliction.utils.error_reporting import LeafflictionError

AUTOMATIC_EXECUTION_MODE = "auto"
SINGLE_IMAGE_EXECUTION_MODE = "image"
DATASET_EXECUTION_MODE = "dataset"

AVAILABLE_EXECUTION_MODES = (
    AUTOMATIC_EXECUTION_MODE,
    SINGLE_IMAGE_EXECUTION_MODE,
    DATASET_EXECUTION_MODE,
)


class AmbiguousExecutionModeError(LeafflictionError):
    """Raised when the requested mode does not match the given path."""


def resolve_execution_mode(requested_execution_mode, target_path):
    """Return the mode to run, deducing it from the path when asked."""
    if requested_execution_mode == AUTOMATIC_EXECUTION_MODE:
        return _deduce_execution_mode_from_path(target_path)
    _raise_when_mode_conflicts_with_path(requested_execution_mode, target_path)
    return requested_execution_mode


def _deduce_execution_mode_from_path(target_path):
    """Return the dataset mode for a directory, the image mode else."""
    if target_path.is_dir():
        return DATASET_EXECUTION_MODE
    return SINGLE_IMAGE_EXECUTION_MODE


def _raise_when_mode_conflicts_with_path(requested_execution_mode, path):
    """Refuse a mode that cannot be applied to the given path."""
    if requested_execution_mode == DATASET_EXECUTION_MODE:
        if not path.is_dir():
            raise AmbiguousExecutionModeError(
                "the dataset mode needs a directory, got: " + str(path)
            )
    if requested_execution_mode == SINGLE_IMAGE_EXECUTION_MODE:
        if path.is_dir():
            raise AmbiguousExecutionModeError(
                "the image mode needs an image file, got: " + str(path)
            )
