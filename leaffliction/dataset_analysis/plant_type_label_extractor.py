"""Derive the plant label used to title the distribution charts."""

CHART_TITLE_SUFFIX = " class distribution"


def extract_plant_type_label_from_directory_path(directory_path):
    """Return the analysed directory name, used as the plant label."""
    return directory_path.name


def build_chart_title_from_plant_type_label(plant_type_label):
    """Return the figure title, for instance "apple class distribution"."""
    return plant_type_label.lower() + CHART_TITLE_SUFFIX
