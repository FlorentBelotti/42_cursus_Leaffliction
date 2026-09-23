"""Render a class distribution as a pie chart and a bar chart."""

from matplotlib import pyplot

from leaffliction.dataset_analysis.categorical_color_palette import (
    build_ordered_color_list,
)
from leaffliction.utils.chart_style import AXES_TITLE_FONT_SIZE
from leaffliction.utils.chart_style import BASELINE_COLOR
from leaffliction.utils.chart_style import CHART_SURFACE_COLOR
from leaffliction.utils.chart_style import FIGURE_TITLE_FONT_SIZE
from leaffliction.utils.chart_style import GRIDLINE_COLOR
from leaffliction.utils.chart_style import LABEL_FONT_SIZE
from leaffliction.utils.chart_style import MUTED_INK_COLOR
from leaffliction.utils.chart_style import PRIMARY_INK_COLOR
from leaffliction.utils.chart_style import SECONDARY_INK_COLOR
from leaffliction.utils.chart_style import SURFACE_GAP_LINE_WIDTH
from leaffliction.utils.chart_style import VALUE_FONT_SIZE

FIGURE_WIDTH_IN_INCHES = 13.0
FIGURE_HEIGHT_IN_INCHES = 5.6
SAVED_FIGURE_RESOLUTION_IN_DOTS_PER_INCH = 150

PIE_AXES_TITLE = "share of each class"
BAR_AXES_TITLE = "images per class"
BAR_WIDTH_RATIO = 0.68
PIE_START_ANGLE_IN_DEGREES = 90


class DistributionChartRenderer:
    """Build the matplotlib figure describing a class distribution."""

    def __init__(self, chart_title):
        """Store the title printed above the two charts."""
        self.chart_title = chart_title

    def render_distribution_figure(self, class_distribution):
        """Return a figure holding the pie chart and the bar chart."""
        class_names = (
            class_distribution.get_class_names_sorted_alphabetically()
        )
        image_counts = (
            class_distribution.get_image_counts_in_class_name_order()
        )
        ordered_colors = build_ordered_color_list(class_names)
        figure, axes_pair = pyplot.subplots(
            nrows=1,
            ncols=2,
            figsize=(FIGURE_WIDTH_IN_INCHES, FIGURE_HEIGHT_IN_INCHES),
        )
        self._apply_figure_chrome(figure)
        self._draw_pie_chart_on_axes(
            axes_pair[0],
            class_names,
            image_counts,
            ordered_colors,
        )
        self._draw_bar_chart_on_axes(
            axes_pair[1],
            class_names,
            image_counts,
            ordered_colors,
        )
        figure.tight_layout()
        return figure

    def save_figure_to_file(self, figure, output_image_path):
        """Write the figure to an image file on the disk."""
        output_image_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(
            output_image_path,
            dpi=SAVED_FIGURE_RESOLUTION_IN_DOTS_PER_INCH,
            facecolor=CHART_SURFACE_COLOR,
        )

    def _apply_figure_chrome(self, figure):
        """Paint the figure background and print the main title."""
        figure.patch.set_facecolor(CHART_SURFACE_COLOR)
        figure.suptitle(
            self.chart_title,
            color=PRIMARY_INK_COLOR,
            fontsize=FIGURE_TITLE_FONT_SIZE,
        )

    def _draw_pie_chart_on_axes(
        self,
        pie_axes,
        class_names,
        image_counts,
        ordered_colors,
    ):
        """Draw the share of every class as a pie chart."""
        pie_axes.set_facecolor(CHART_SURFACE_COLOR)
        pie_axes.pie(
            image_counts,
            labels=class_names,
            colors=ordered_colors,
            autopct="%1.1f%%",
            startangle=PIE_START_ANGLE_IN_DEGREES,
            counterclock=False,
            wedgeprops=self._build_pie_wedge_properties(),
            textprops={
                "color": SECONDARY_INK_COLOR,
                "fontsize": LABEL_FONT_SIZE,
            },
        )
        pie_axes.set_title(
            PIE_AXES_TITLE,
            color=SECONDARY_INK_COLOR,
            fontsize=AXES_TITLE_FONT_SIZE,
        )

    def _build_pie_wedge_properties(self):
        """Return the wedge style separating slices with a surface gap."""
        return {
            "edgecolor": CHART_SURFACE_COLOR,
            "linewidth": SURFACE_GAP_LINE_WIDTH,
        }

    def _draw_bar_chart_on_axes(
        self,
        bar_axes,
        class_names,
        image_counts,
        ordered_colors,
    ):
        """Draw the absolute image count of every class as bars."""
        bar_axes.set_facecolor(CHART_SURFACE_COLOR)
        bar_container = bar_axes.bar(
            class_names,
            image_counts,
            color=ordered_colors,
            edgecolor=CHART_SURFACE_COLOR,
            linewidth=SURFACE_GAP_LINE_WIDTH,
            width=BAR_WIDTH_RATIO,
        )
        bar_axes.bar_label(
            bar_container,
            padding=3,
            color=SECONDARY_INK_COLOR,
            fontsize=VALUE_FONT_SIZE,
        )
        bar_axes.set_title(
            BAR_AXES_TITLE,
            color=SECONDARY_INK_COLOR,
            fontsize=AXES_TITLE_FONT_SIZE,
        )
        self._apply_recessive_bar_axes_chrome(bar_axes)

    def _apply_recessive_bar_axes_chrome(self, bar_axes):
        """Push the grid, the spines and the ticks behind the data."""
        bar_axes.set_axisbelow(True)
        bar_axes.yaxis.grid(
            True,
            color=GRIDLINE_COLOR,
            linewidth=0.8,
        )
        bar_axes.xaxis.grid(False)
        self._hide_decorative_spines(bar_axes)
        bar_axes.tick_params(
            axis="both",
            colors=MUTED_INK_COLOR,
            labelsize=LABEL_FONT_SIZE,
            length=0,
        )
        pyplot.setp(
            bar_axes.get_xticklabels(),
            rotation=20,
            horizontalalignment="right",
        )

    def _hide_decorative_spines(self, bar_axes):
        """Keep only the baseline, in the recessive baseline color."""
        for hidden_spine_name in ("top", "right", "left"):
            bar_axes.spines[hidden_spine_name].set_visible(False)
        bar_axes.spines["bottom"].set_color(BASELINE_COLOR)
