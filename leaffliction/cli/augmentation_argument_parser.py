"""Command line interface of the Augmentation entrypoint."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from leaffliction.cli.augmentation_execution_mode import (
    AUTOMATIC_EXECUTION_MODE,
)
from leaffliction.cli.augmentation_execution_mode import (
    AVAILABLE_EXECUTION_MODES,
)
from leaffliction.cli.augmentation_execution_mode import (
    DATASET_EXECUTION_MODE,
)
from leaffliction.cli.augmentation_execution_mode import (
    resolve_execution_mode,
)
from leaffliction.utils.filesystem_path_validator import (
    resolve_writable_path,
)
from leaffliction.utils.filesystem_path_validator import (
    validate_existing_directory,
)
from leaffliction.utils.filesystem_path_validator import (
    validate_existing_image_file,
)

PROGRAM_NAME = "Augmentation.py"
PROGRAM_DESCRIPTION = (
    "Augment leaf images. Given an image file, build and display its "
    "six augmented variants next to the original. Given a directory, "
    "rebuild a balanced copy of that data set."
)
PROGRAM_EPILOG = (
    "examples:\n"
    "  ./Augmentation.py 'Apple/apple_healthy/image (1).JPG'\n"
    "  ./Augmentation.py ./Apple --destination augmented_directory\n"
    "  ./Augmentation.py ./Apple --mode dataset --force\n"
)
DEFAULT_DESTINATION_DIRECTORY_NAME = "augmented_directory"


@dataclass
class AugmentationArguments:
    """Validated arguments of the Augmentation entrypoint."""

    execution_mode: str
    image_file_path: Optional[Path]
    output_directory_path: Optional[Path]
    preview_output_path: Optional[Path]
    source_directory_path: Optional[Path]
    destination_directory_path: Optional[Path]
    should_overwrite_destination: bool
    should_display_figure: bool


def build_augmentation_argument_parser():
    """Return the argument parser of the Augmentation entrypoint."""
    argument_parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description=PROGRAM_DESCRIPTION,
        epilog=PROGRAM_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    argument_parser.add_argument(
        "target_path",
        help="an image file to augment, or a data set directory",
    )
    argument_parser.add_argument(
        "--mode",
        dest="requested_execution_mode",
        choices=AVAILABLE_EXECUTION_MODES,
        default=AUTOMATIC_EXECUTION_MODE,
        help="force the image or dataset mode instead of deducing it",
    )
    _add_single_image_arguments(argument_parser)
    _add_dataset_arguments(argument_parser)
    argument_parser.add_argument(
        "--no-display",
        dest="should_display_figure",
        action="store_false",
        help="never open a window, useful on a headless machine",
    )
    return argument_parser


def _add_single_image_arguments(argument_parser):
    """Declare the options that only the image mode reads."""
    argument_parser.add_argument(
        "-o",
        "--output-directory",
        dest="output_directory_path",
        default=None,
        help="write the variants there instead of beside the original",
    )
    argument_parser.add_argument(
        "--preview-output",
        dest="preview_output_path",
        default=None,
        help="also write the comparison board to this image file",
    )


def _add_dataset_arguments(argument_parser):
    """Declare the options that only the dataset mode reads."""
    argument_parser.add_argument(
        "-d",
        "--destination",
        dest="destination_directory_path",
        default=DEFAULT_DESTINATION_DIRECTORY_NAME,
        help="directory receiving the balanced copy of the data set",
    )
    argument_parser.add_argument(
        "--force",
        dest="should_overwrite_destination",
        action="store_true",
        help="write into the destination even if it already exists",
    )


def parse_augmentation_arguments(raw_command_line_arguments):
    """Parse and validate the Augmentation command line arguments."""
    argument_parser = build_augmentation_argument_parser()
    parsed_arguments = argument_parser.parse_args(raw_command_line_arguments)
    execution_mode = resolve_execution_mode(
        parsed_arguments.requested_execution_mode,
        Path(parsed_arguments.target_path).expanduser(),
    )
    if execution_mode == DATASET_EXECUTION_MODE:
        return _build_dataset_arguments(parsed_arguments, execution_mode)
    return _build_single_image_arguments(parsed_arguments, execution_mode)


def _build_dataset_arguments(parsed_arguments, execution_mode):
    """Return the arguments needed to balance a whole data set."""
    return AugmentationArguments(
        execution_mode=execution_mode,
        image_file_path=None,
        output_directory_path=None,
        preview_output_path=None,
        source_directory_path=validate_existing_directory(
            parsed_arguments.target_path
        ),
        destination_directory_path=resolve_writable_path(
            parsed_arguments.destination_directory_path
        ),
        should_overwrite_destination=(
            parsed_arguments.should_overwrite_destination
        ),
        should_display_figure=parsed_arguments.should_display_figure,
    )


def _build_single_image_arguments(parsed_arguments, execution_mode):
    """Return the arguments needed to augment a single image."""
    return AugmentationArguments(
        execution_mode=execution_mode,
        image_file_path=validate_existing_image_file(
            parsed_arguments.target_path
        ),
        output_directory_path=_resolve_optional_path(
            parsed_arguments.output_directory_path
        ),
        preview_output_path=_resolve_optional_path(
            parsed_arguments.preview_output_path
        ),
        source_directory_path=None,
        destination_directory_path=None,
        should_overwrite_destination=False,
        should_display_figure=parsed_arguments.should_display_figure,
    )


def _resolve_optional_path(raw_path):
    """Return the resolved path, or None when the option was omitted."""
    if raw_path is None:
        return None
    return resolve_writable_path(raw_path)
