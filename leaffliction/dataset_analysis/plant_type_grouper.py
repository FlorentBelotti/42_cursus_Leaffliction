"""Split the classes of a data set by plant type, one chart per plant.

The subject asks for pie and bar charts "for each plant type". Two
layouts of the data set are handled:

- nested, ``images/Apple/Apple_healthy``: the plant is the directory
  holding the class directories;
- flat, ``images/Apple_healthy``: the plant is the prefix of the class
  name, before the first underscore.

When neither rule finds several plants, every class is drawn on one
single chart titled after the analysed directory, which is exactly
the ``./Distribution.py ./Apple`` case of the subject.
"""

CLASS_NAME_PLANT_SEPARATOR = "_"


def group_image_paths_by_plant_type(grouped_image_paths, source_directory):
    """Return a mapping of plant label to its own class to paths mapping."""
    grouped_by_directory = _group_by_plant_directory(
        grouped_image_paths,
        source_directory,
    )
    if len(grouped_by_directory) > 1:
        return grouped_by_directory
    grouped_by_prefix = _group_by_class_name_prefix(grouped_image_paths)
    if _is_convincing_prefix_grouping(grouped_by_prefix):
        return grouped_by_prefix
    return {source_directory.name: grouped_image_paths}


def _group_by_plant_directory(grouped_image_paths, source_directory):
    """Group the classes by the first directory under the source."""
    grouped_by_directory = {}
    for class_name in grouped_image_paths:
        class_directory = grouped_image_paths[class_name][0].parent
        plant_label = _extract_plant_directory_name(
            class_directory,
            source_directory,
        )
        _append_class_to_group(
            grouped_by_directory,
            plant_label,
            class_name,
            grouped_image_paths[class_name],
        )
    return grouped_by_directory


def _extract_plant_directory_name(class_directory, source_directory):
    """Return the first path component of the class under the source."""
    relative_parts = class_directory.relative_to(source_directory).parts
    if len(relative_parts) < 2:
        return source_directory.name
    return relative_parts[0]


def _group_by_class_name_prefix(grouped_image_paths):
    """Group the classes by the word before the first underscore."""
    grouped_by_prefix = {}
    for class_name in grouped_image_paths:
        plant_label = class_name.split(CLASS_NAME_PLANT_SEPARATOR)[0]
        _append_class_to_group(
            grouped_by_prefix,
            plant_label,
            class_name,
            grouped_image_paths[class_name],
        )
    return grouped_by_prefix


def _is_convincing_prefix_grouping(grouped_by_prefix):
    """Accept the prefix rule only when it yields real plant groups."""
    if len(grouped_by_prefix) < 2:
        return False
    for plant_label in grouped_by_prefix:
        if len(grouped_by_prefix[plant_label]) < 2:
            return False
    return True


def _append_class_to_group(groups, plant_label, class_name, image_paths):
    """Store one class under its plant label, creating the group."""
    if plant_label not in groups:
        groups[plant_label] = {}
    groups[plant_label][class_name] = image_paths
