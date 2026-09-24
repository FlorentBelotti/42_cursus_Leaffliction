"""Render the original leaf image next to its six augmented variants."""

from matplotlib import pyplot

from leaffliction.utils.chart_style import AXES_TITLE_FONT_SIZE
from leaffliction.utils.chart_style import CHART_SURFACE_COLOR
from leaffliction.utils.chart_style import SECONDARY_INK_COLOR

ORIGINAL_IMAGE_LABEL = "Original"
PREVIEW_CELL_WIDTH_IN_INCHES = 2.1
PREVIEW_FIGURE_HEIGHT_IN_INCHES = 2.9
SAVED_FIGURE_RESOLUTION_IN_DOTS_PER_INCH = 150


class AugmentationPreviewRenderer:
    """Build the single row figure showing every augmentation."""

    def render_preview_figure(
        self,
        original_image,
        augmented_images_by_name,
    ):
        """Return a figure with the original image and each variant."""
        labeled_images = self._build_labeled_image_sequence(
            original_image,
            augmented_images_by_name,
        )
        figure, axes_row = pyplot.subplots(
            nrows=1,
            ncols=len(labeled_images),
            figsize=self._compute_figure_size(len(labeled_images)),
        )
        figure.patch.set_facecolor(CHART_SURFACE_COLOR)
        for cell_index in range(len(labeled_images)):
            image_label, image_to_draw = labeled_images[cell_index]
            self._draw_labeled_image_on_axes(
                axes_row[cell_index],
                image_to_draw,
                image_label,
            )
        figure.tight_layout()
        return figure

    def save_figure_to_file(self, figure, output_image_path):
        """Write the preview figure to an image file on the disk."""
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_image_path,
            dpi=SAVED_FIGURE_RESOLUTION_IN_DOTS_PER_INCH,
            facecolor=CHART_SURFACE_COLOR,
        )

    def _build_labeled_image_sequence(
        self,
        original_image,
        augmented_images_by_name,
    ):
        """Return label and image pairs, original first then variants."""
        labeled_images = [(ORIGINAL_IMAGE_LABEL, original_image)]
        for augmentation_name in augmented_images_by_name:
            labeled_images.append(
                (
                    augmentation_name,
                    augmented_images_by_name[augmentation_name],
                )
            )
        return labeled_images

    def _compute_figure_size(self, cell_count):
        """Return the figure size fitting the requested cell count."""
        figure_width = PREVIEW_CELL_WIDTH_IN_INCHES * cell_count
        return (figure_width, PREVIEW_FIGURE_HEIGHT_IN_INCHES)

    def _draw_labeled_image_on_axes(self, axes, image_to_draw, image_label):
        """Draw one image with its label, without any axis decoration."""
        axes.imshow(image_to_draw)
        axes.set_title(
            image_label,
            color=SECONDARY_INK_COLOR,
            fontsize=AXES_TITLE_FONT_SIZE,
        )
        axes.set_axis_off()
