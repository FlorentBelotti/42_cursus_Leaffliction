#!/usr/bin/env python3
"""Entrypoint of part 2: augmentation of leaf images.

Usage:
    ./Augmentation.py 'Apple/apple_healthy/image (1).JPG'
    ./Augmentation.py ./Apple --destination augmented_directory
    ./Augmentation.py -h

Given an image file the program writes the six augmented variants next
to the original and displays them side by side. Given a directory it
rebuilds a balanced copy of the whole data set. The mode is deduced
from the path, and can be forced with the mode option.
"""

import sys

from leaffliction.cli.augmentation_argument_parser import (
    parse_augmentation_arguments,
)
from leaffliction.cli.augmentation_execution_mode import (
    DATASET_EXECUTION_MODE,
)
from leaffliction.dataset_balancing.dataset_balancing_pipeline import (
    run_dataset_balancing,
)
from leaffliction.image_augmentation.single_image_pipeline import (
    run_single_image_augmentation,
)
from leaffliction.utils.error_reporting import LeafflictionError
from leaffliction.utils.error_reporting import report_error_and_exit


def main(raw_command_line_arguments):
    """Parse the arguments then run the requested augmentation mode."""
    try:
        arguments = parse_augmentation_arguments(raw_command_line_arguments)
        if arguments.execution_mode == DATASET_EXECUTION_MODE:
            run_whole_dataset_mode(arguments)
        else:
            run_single_image_mode(arguments)
    except LeafflictionError as raised_error:
        report_error_and_exit(str(raised_error))


def run_single_image_mode(arguments):
    """Augment one image, save the variants and plot the board."""
    run_single_image_augmentation(
        arguments.image_file_path,
        arguments.output_directory_path,
        arguments.preview_output_path,
        arguments.should_display_figure,
    )


def run_whole_dataset_mode(arguments):
    """Rebuild a balanced copy of the whole data set."""
    run_dataset_balancing(
        arguments.source_directory_path,
        arguments.destination_directory_path,
        arguments.should_overwrite_destination,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
