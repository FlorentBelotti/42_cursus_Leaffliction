#!/usr/bin/env python3
"""Entrypoint rebuilding a balanced copy of a leaf image data set.

Usage:
    ./balance_dataset.py ./Apple --destination augmented_directory

This is the same work as ./Augmentation.py ./Apple, exposed under an
explicit name. Both entrypoints call the very same pipeline, so there
is exactly one implementation of the balancing logic.
"""

import sys

from leaffliction.cli.balance_dataset_argument_parser import (
    parse_balance_dataset_arguments,
)
from leaffliction.dataset_balancing.dataset_balancing_pipeline import (
    run_dataset_balancing,
)
from leaffliction.utils.error_reporting import LeafflictionError
from leaffliction.utils.error_reporting import report_error_and_exit


def main(raw_command_line_arguments):
    """Parse the arguments then balance the given data set."""
    try:
        arguments = parse_balance_dataset_arguments(
            raw_command_line_arguments
        )
        run_dataset_balancing(
            arguments.source_directory_path,
            arguments.destination_directory_path,
            arguments.should_overwrite_destination,
        )
    except LeafflictionError as raised_error:
        report_error_and_exit(str(raised_error))


if __name__ == "__main__":
    main(sys.argv[1:])
