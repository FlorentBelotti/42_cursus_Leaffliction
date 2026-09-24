"""Chronological orchestration of the single image augmentation."""

from leaffliction.image_augmentation.augmentation_preview_renderer import (
    AugmentationPreviewRenderer,
)
from leaffliction.image_augmentation.augmentation_registry import (
    AugmentationRegistry,
)
from leaffliction.image_augmentation.augmented_image_writer import (
    AugmentedImageWriter,
)
from leaffliction.utils.figure_display import (
    display_figure_when_display_is_available,
)
from leaffliction.utils.image_file_loader import load_image_as_rgb


def run_single_image_augmentation(
    image_file_path,
    output_directory_path,
    preview_output_path,
    should_display_figure,
):
    """Augment one image six times, save the files and plot the board."""
    original_image = load_image_as_rgb(image_file_path)
    augmented_images_by_name = apply_every_augmentation_to_image(
        original_image
    )
    write_every_augmented_image(
        augmented_images_by_name,
        image_file_path,
        output_directory_path,
    )
    preview_renderer = AugmentationPreviewRenderer()
    preview_figure = preview_renderer.render_preview_figure(
        original_image,
        augmented_images_by_name,
    )
    if preview_output_path is not None:
        preview_renderer.save_figure_to_file(
            preview_figure,
            preview_output_path,
        )
        print("Preview written to: " + str(preview_output_path))
    if should_display_figure:
        display_figure_when_display_is_available(preview_figure)


def apply_every_augmentation_to_image(original_image):
    """Return every augmented variant of the image, keyed by name."""
    augmentation_registry = AugmentationRegistry()
    augmentation_sequence = (
        augmentation_registry.build_default_augmentation_sequence()
    )
    augmented_images_by_name = {}
    for augmentation in augmentation_sequence:
        augmented_images_by_name[augmentation.augmentation_name] = (
            augmentation.apply_to_image(original_image)
        )
    return augmented_images_by_name


def write_every_augmented_image(
    augmented_images_by_name,
    original_image_path,
    output_directory_path,
):
    """Write every variant next to the original, or in the given folder."""
    augmented_image_writer = AugmentedImageWriter(output_directory_path)
    for augmentation_name in augmented_images_by_name:
        written_image_path = augmented_image_writer.write_augmented_image(
            augmented_images_by_name[augmentation_name],
            original_image_path,
            augmentation_name,
        )
        print("Saved: " + str(written_image_path))
