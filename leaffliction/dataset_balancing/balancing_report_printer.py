"""Print what the balancing stage did, class by class."""

REPORT_HEADER_LINE = "Balancing report"
REPORT_SEPARATOR_LINE = "-" * 62


def print_balancing_report(balancing_plans, destination_directory_path):
    """Print one line per class, then where the copy was written."""
    print(REPORT_HEADER_LINE)
    print(REPORT_SEPARATOR_LINE)
    label_column_width = _compute_label_column_width(balancing_plans)
    for balancing_plan in balancing_plans:
        print(_build_plan_report_line(balancing_plan, label_column_width))
    print(REPORT_SEPARATOR_LINE)
    print("Augmented data set written to: " + str(destination_directory_path))


def _build_plan_report_line(balancing_plan, label_column_width):
    """Return the report line describing one balanced class."""
    return (
        balancing_plan.class_name.ljust(label_column_width)
        + str(balancing_plan.existing_image_count).rjust(7)
        + " + "
        + str(balancing_plan.images_to_generate_count).rjust(6)
        + " = "
        + str(balancing_plan.get_target_image_count()).rjust(7)
    )


def _compute_label_column_width(balancing_plans):
    """Return the width needed to align every class name."""
    longest_class_name_length = 0
    for balancing_plan in balancing_plans:
        if len(balancing_plan.class_name) > longest_class_name_length:
            longest_class_name_length = len(balancing_plan.class_name)
    return longest_class_name_length + 2
