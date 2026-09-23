#!/usr/bin/env python3
"""Entrypoint of part 1: analysis of a leaf image data set.

Usage:
    ./Distribution.py ./Apple
    ./Distribution.py ./Apple --output charts/apple.png --no-display
"""

import sys

from leaffliction.cli.distribution_argument_parser import (
    parse_distribution_arguments,
)
from leaffliction.dataset_analysis.dataset_distribution_pipeline import (
    run_dataset_distribution_analysis,
)
from leaffliction.utils.error_reporting import LeafflictionError
from leaffliction.utils.error_reporting import report_error_and_exit


def main(raw_command_line_arguments):
    """Parse the arguments then run the distribution analysis."""
    try:
        arguments = parse_distribution_arguments(raw_command_line_arguments)
        run_dataset_distribution_analysis(
            arguments.source_directory_path,
            arguments.output_image_path,
            arguments.should_display_figure,
        )
    except LeafflictionError as raised_error:
        report_error_and_exit(str(raised_error))


if __name__ == "__main__":
    main(sys.argv[1:])
