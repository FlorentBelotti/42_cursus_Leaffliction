"""Derive the title of the distribution charts from a plant label."""

CHART_TITLE_SUFFIX = " class distribution"


def build_chart_title_from_plant_type_label(plant_type_label):
    """Return the figure title, for instance "apple class distribution"."""
    return plant_type_label.lower() + CHART_TITLE_SUFFIX
