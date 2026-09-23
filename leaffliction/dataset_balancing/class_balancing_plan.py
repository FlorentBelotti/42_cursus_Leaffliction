"""How many images one class still needs to reach the target size."""

from dataclasses import dataclass


@dataclass
class ClassBalancingPlan:
    """Describe the work needed to bring one class to the target size."""

    class_name: str
    existing_image_count: int
    images_to_generate_count: int

    def get_target_image_count(self):
        """Return the class size once the plan has been carried out."""
        return self.existing_image_count + self.images_to_generate_count

    def is_already_balanced(self):
        """Return True when the class needs no extra image at all."""
        return self.images_to_generate_count == 0
