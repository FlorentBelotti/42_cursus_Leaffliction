"""Generate the augmented images that bring a class to its target size."""

from leaffliction.dataset_balancing.generation_step import GenerationStep
from leaffliction.image_augmentation.augmented_image_writer import (
    AugmentedImageWriter,
)
from leaffliction.utils.image_file_loader import load_image_as_rgb


class DatasetBalancer:
    """Fill a class directory with deterministic augmented images.

    The pairs image and augmentation are walked in a fixed round robin:
    the source image changes at every step, and the augmentation only
    changes once every source image has been used. Two consequences
    matter. Every image of the class is augmented before any image is
    augmented twice, so the added diversity is spread evenly. And the
    whole augmented data set is reproducible, since no random draw is
    involved anywhere.
    """

    def __init__(self, augmentation_registry):
        """Store the ordered augmentation sequence used by the walk."""
        self.augmentation_sequence = (
            augmentation_registry.build_default_augmentation_sequence()
        )

    def generate_missing_images_for_class(
        self,
        class_balancing_plan,
        class_image_paths,
        destination_class_directory_path,
    ):
        """Write the images needed to reach the target size of a class."""
        augmented_image_writer = AugmentedImageWriter(
            destination_class_directory_path
        )
        written_image_paths = []
        for step_index in range(
            class_balancing_plan.images_to_generate_count
        ):
            generation_step = self._build_generation_step(
                step_index,
                class_image_paths,
            )
            written_image_paths.append(
                self._carry_out_generation_step(
                    generation_step,
                    augmented_image_writer,
                )
            )
        return written_image_paths

    def _build_generation_step(self, step_index, class_image_paths):
        """Return which image to augment and how, for that step."""
        image_count = len(class_image_paths)
        source_image_path = class_image_paths[step_index % image_count]
        variation_index = self._compute_variation_index(
            step_index,
            image_count,
        )
        base_augmentation = self._select_base_augmentation(
            step_index,
            image_count,
        )
        return GenerationStep(
            source_image_path=source_image_path,
            augmentation=base_augmentation.build_variant_with_index(
                variation_index
            ),
            variation_index=variation_index,
        )

    def _select_base_augmentation(self, step_index, image_count):
        """Return the augmentation of that step of the round robin."""
        augmentation_index = (step_index // image_count) % len(
            self.augmentation_sequence
        )
        return self.augmentation_sequence[augmentation_index]

    def _compute_variation_index(self, step_index, image_count):
        """Return how many full cycles of the round robin are done."""
        steps_per_full_cycle = image_count * len(self.augmentation_sequence)
        return step_index // steps_per_full_cycle

    def _carry_out_generation_step(
        self,
        generation_step,
        augmented_image_writer,
    ):
        """Load, augment and write the image described by the step."""
        source_image = load_image_as_rgb(generation_step.source_image_path)
        augmented_image = generation_step.augmentation.apply_to_image(
            source_image
        )
        return augmented_image_writer.write_augmented_image(
            augmented_image,
            generation_step.source_image_path,
            generation_step.augmentation.augmentation_name,
            generation_step.variation_index,
        )
