"""Fixed categorical palette shared by every chart of the project.

The eight hues are assigned in a fixed order, one per class name taken
in alphabetical order. Because the assignment depends only on the class
names, a class keeps the same color in the pie chart, in the bar chart,
and across two runs of the program.

The order was chosen so that adjacent slots stay separable for the most
common colour vision deficiencies, which matters here because the pie
chart puts adjacent slots side by side.
"""

CATEGORICAL_COLOR_SEQUENCE = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)

FALLBACK_COLOR_FOR_EXTRA_CLASSES = "#898781"


def build_color_per_class_name(class_names):
    """Map every class name to a stable color of the palette."""
    color_per_class_name = {}
    for class_index, class_name in enumerate(class_names):
        color_per_class_name[class_name] = _select_color_for_slot(class_index)
    return color_per_class_name


def build_ordered_color_list(class_names):
    """Return the palette colors aligned on the given class name list."""
    color_per_class_name = build_color_per_class_name(class_names)
    ordered_colors = []
    for class_name in class_names:
        ordered_colors.append(color_per_class_name[class_name])
    return ordered_colors


def _select_color_for_slot(class_index):
    """Return the color of that slot, gray beyond the eighth slot."""
    if class_index < len(CATEGORICAL_COLOR_SEQUENCE):
        return CATEGORICAL_COLOR_SEQUENCE[class_index]
    return FALLBACK_COLOR_FOR_EXTRA_CLASSES
