"""Command line interface of the Distribution entrypoint."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from leaffliction.utils.filesystem_path_validator import (
    resolve_writable_path,
)
from leaffliction.utils.filesystem_path_validator import (
    validate_existing_directory,
)

PROGRAM_NAME = "Distribution.py"
PROGRAM_DESCRIPTION = (
    "Analyse a leaf image data set and plot how many images each "
    "class holds, as a pie chart and a bar chart."
)


@dataclass
class DistributionArguments:
    """Validated arguments of the Distribution entrypoint."""

    source_directory_path: Path
    output_image_path: Optional[Path]
    should_display_figure: bool


def build_distribution_argument_parser():
    """Return the argument parser of the Distribution entrypoint."""
    argument_parser = argparse.ArgumentParser(
        prog=PROGRAM_NAME,
        description=PROGRAM_DESCRIPTION,
    )
    argument_parser.add_argument(
        "source_directory",
        help="directory holding the class subdirectories to analyse",
    )
    argument_parser.add_argument(
        "-o",
        "--output",
        dest="output_image_path",
        default=None,
        help="also write the charts to this image file",
    )
    argument_parser.add_argument(
        "--no-display",
        dest="should_display_figure",
        action="store_false",
        help="never open a window, useful on a headless machine",
    )
    return argument_parser


def parse_distribution_arguments(raw_command_line_arguments):
    """Parse and validate the Distribution command line arguments."""
    argument_parser = build_distribution_argument_parser()
    parsed_arguments = argument_parser.parse_args(raw_command_line_arguments)
    source_directory_path = validate_existing_directory(
        parsed_arguments.source_directory
    )
    output_image_path = _resolve_optional_output_path(
        parsed_arguments.output_image_path
    )
    return DistributionArguments(
        source_directory_path=source_directory_path,
        output_image_path=output_image_path,
        should_display_figure=parsed_arguments.should_display_figure,
    )


def _resolve_optional_output_path(raw_output_image_path):
    """Return the resolved output path, or None when not requested."""
    if raw_output_image_path is None:
        return None
    return resolve_writable_path(raw_output_image_path)
