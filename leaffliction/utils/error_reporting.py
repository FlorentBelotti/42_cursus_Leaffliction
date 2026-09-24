"""Error types and terminal reporting shared by every entrypoint.

Every failure the project can anticipate is raised as a
``LeafflictionError`` subclass, so that the entrypoints catch a single
family of exceptions and print a readable message instead of leaking a
traceback to the evaluator.
"""

import sys


class LeafflictionError(Exception):
    """Base class of every error raised deliberately by the project."""


class InvalidDirectoryError(LeafflictionError):
    """Raised when a directory argument cannot be used."""


class InvalidImageFileError(LeafflictionError):
    """Raised when an image argument cannot be read as an image."""


class EmptyDataSetError(LeafflictionError):
    """Raised when a directory holds no supported image file."""


class DestinationAlreadyExistsError(LeafflictionError):
    """Raised when the augmented data set directory already exists."""


def report_error_and_exit(error_message):
    """Print the message on the error stream and exit with a failure."""
    print("Error: " + error_message, file=sys.stderr)
    sys.exit(1)
