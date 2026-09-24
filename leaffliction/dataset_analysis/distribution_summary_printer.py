"""Print a textual summary of a class distribution on the terminal."""

SUMMARY_HEADER_LINE = "Class distribution"
SUMMARY_SEPARATOR_LINE = "-" * 46


def print_distribution_summary(class_distribution):
    """Print one line per class, then the totals of the data set."""
    print(SUMMARY_HEADER_LINE)
    print(SUMMARY_SEPARATOR_LINE)
    _print_one_line_per_class(class_distribution)
    print(SUMMARY_SEPARATOR_LINE)
    _print_data_set_totals(class_distribution)


def _print_one_line_per_class(class_distribution):
    """Print the image count of every class, aligned in a column."""
    class_names = class_distribution.get_class_names_sorted_alphabetically()
    label_column_width = _compute_label_column_width(class_names)
    for class_name in class_names:
        image_count = class_distribution.get_image_count_for_class(class_name)
        print(
            class_name.ljust(label_column_width)
            + str(image_count).rjust(7)
        )


def _print_data_set_totals(class_distribution):
    """Print the class count, the image count and the balance gap."""
    print("classes: " + str(class_distribution.get_class_count()))
    print("images: " + str(class_distribution.get_total_image_count()))
    print(
        "smallest class: "
        + str(class_distribution.get_smallest_class_image_count())
        + ", largest class: "
        + str(class_distribution.get_largest_class_image_count())
    )


def _compute_label_column_width(class_names):
    """Return the width needed to align every class name."""
    longest_class_name_length = 0
    for class_name in class_names:
        if len(class_name) > longest_class_name_length:
            longest_class_name_length = len(class_name)
    return longest_class_name_length + 2
