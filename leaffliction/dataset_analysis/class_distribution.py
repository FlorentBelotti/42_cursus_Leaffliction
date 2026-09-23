"""Read only view of how many images every class of a data set holds."""


class ClassDistribution:
    """Hold the image count of every class of a data set."""

    def __init__(self, image_count_per_class):
        """Store a private copy of the class name to count mapping."""
        self.image_count_per_class = dict(image_count_per_class)

    @classmethod
    def from_grouped_image_paths(cls, grouped_image_paths):
        """Build a distribution from the image file collector output."""
        image_count_per_class = {}
        for class_name in grouped_image_paths:
            image_count_per_class[class_name] = len(
                grouped_image_paths[class_name]
            )
        return cls(image_count_per_class)

    def get_class_names_sorted_alphabetically(self):
        """Return the class names in a stable alphabetical order."""
        return sorted(self.image_count_per_class.keys())

    def get_image_count_for_class(self, class_name):
        """Return how many images the given class holds."""
        return self.image_count_per_class[class_name]

    def get_image_counts_in_class_name_order(self):
        """Return the counts aligned on the sorted class name list."""
        ordered_image_counts = []
        for class_name in self.get_class_names_sorted_alphabetically():
            ordered_image_counts.append(
                self.image_count_per_class[class_name]
            )
        return ordered_image_counts

    def get_class_count(self):
        """Return how many distinct classes the data set holds."""
        return len(self.image_count_per_class)

    def get_largest_class_image_count(self):
        """Return the image count of the most populated class."""
        return max(self.image_count_per_class.values())

    def get_smallest_class_image_count(self):
        """Return the image count of the least populated class."""
        return min(self.image_count_per_class.values())

    def get_total_image_count(self):
        """Return how many images the whole data set holds."""
        return sum(self.image_count_per_class.values())
