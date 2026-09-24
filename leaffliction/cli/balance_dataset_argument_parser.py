"""Command line interface of the balance_dataset entrypoint."""

import argparse
from dataclasses import dataclass
from pathlib import Path

from leaffliction.cli.augmentation_argument_parser import (
    DEFAULT_DESTINATION_DIRECTORY_NAME,
)
from leaffliction.utils.filesystem_path_validator import (
    resolve_writable_path,
)
from leaffliction.utils.filesystem_path_validator import (
    validate_existing_directory,
)

PROGRAM_NAME = "balance_dataset.py"
PROGRAM_DESCRIPTION = (
    "Rebuild a balanced copy of a leaf image data set, by generating "
    "augmented images until every class holds as many images as the "
    "most populated one."
)


@dataclass
class BalanceDatasetArguments:
    """Validated arguments of the balance_dataset entrypoint."""

    source_directory_path: Path
    destination_directory_path: Path
    should_overwrite_destination: bool


def build_balance_dataset_argument_parser():
    """Return the argument parser of the balance_dataset entrypoint."""
    argument_parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description=PROGRAM_DESCRIPTION,
    )
    argument_parser.add_argument(
        "source_directory",
        help="data set directory holding the class subdirectories",
    )
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
    return argument_parser


def parse_balance_dataset_arguments(raw_command_line_arguments):
    """Parse and validate the balance_dataset command line arguments."""
    argument_parser = build_balance_dataset_argument_parser()
    parsed_arguments = argument_parser.parse_args(raw_command_line_arguments)
    return BalanceDatasetArguments(
        source_directory_path=validate_existing_directory(
            parsed_arguments.source_directory
        ),
        destination_directory_path=resolve_writable_path(
            parsed_arguments.destination_directory_path
        ),
        should_overwrite_destination=(
            parsed_arguments.should_overwrite_destination
        ),
    )
