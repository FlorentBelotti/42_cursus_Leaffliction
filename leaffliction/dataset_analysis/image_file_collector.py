"""Recursive collection of the data set images, grouped by class."""

from leaffliction.utils.error_reporting import EmptyDataSetError
from leaffliction.utils.supported_image_extensions import (
    is_supported_image_file,
)


class ImageFileCollector:
    """Walk a data set directory and group its images by class name.

    The class of an image is the name of the directory that directly
    contains it. That rule makes the collector independent from the
    depth of the directory given on the command line: pointing it at
    "./Apple" yields the four apple classes, and pointing it at the data
    set root yields every class of every plant.
    """

    def __init__(self, root_directory_path):
        """Store the root directory the collector will walk."""
        self.root_directory_path = root_directory_path

    def collect_image_paths_grouped_by_class(self):
        """Return a mapping of every class name to its image paths."""
        grouped_image_paths = {}
        for image_file_path in self._iterate_supported_image_files():
            class_name = self._extract_class_name_from_image_path(
                image_file_path
            )
            if class_name not in grouped_image_paths:
                grouped_image_paths[class_name] = []
            grouped_image_paths[class_name].append(image_file_path)
        self._raise_when_no_image_was_found(grouped_image_paths)
        return self._sort_every_class_of_the_mapping(grouped_image_paths)

    def _iterate_supported_image_files(self):
        """Yield every supported image file found under the root."""
        for candidate_path in self.root_directory_path.rglob("*"):
            if is_supported_image_file(candidate_path):
                yield candidate_path

    def _extract_class_name_from_image_path(self, image_file_path):
        """Return the name of the directory directly holding the image."""
        return image_file_path.parent.name

    def _raise_when_no_image_was_found(self, grouped_image_paths):
        """Fail explicitly when the directory holds no usable image."""
        if len(grouped_image_paths) == 0:
            raise EmptyDataSetError(
                "no supported image found under: "
                + str(self.root_directory_path)
            )

    def _sort_every_class_of_the_mapping(self, grouped_image_paths):
        """Return the mapping with class names and paths in stable order.

        Sorting is what makes the whole augmentation stage reproducible:
        the file system iteration order is not guaranteed, but the
        balancing round robin must always pick the same images.
        """
        sorted_grouped_image_paths = {}
        for class_name in sorted(grouped_image_paths.keys()):
            sorted_grouped_image_paths[class_name] = sorted(
                grouped_image_paths[class_name]
            )
        return sorted_grouped_image_paths
