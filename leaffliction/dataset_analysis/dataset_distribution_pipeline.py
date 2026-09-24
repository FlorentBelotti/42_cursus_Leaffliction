"""Chronological orchestration of the data set distribution analysis."""

from leaffliction.dataset_analysis.class_distribution import ClassDistribution
from leaffliction.dataset_analysis.distribution_chart_renderer import (
    DistributionChartRenderer,
)
from leaffliction.dataset_analysis.distribution_summary_printer import (
    print_distribution_summary,
)
from leaffliction.dataset_analysis.image_file_collector import (
    ImageFileCollector,
)
from leaffliction.dataset_analysis.plant_type_label_extractor import (
    build_chart_title_from_plant_type_label,
)
from leaffliction.dataset_analysis.plant_type_label_extractor import (
    extract_plant_type_label_from_directory_path,
)
from leaffliction.utils.figure_display import (
    display_figure_when_display_is_available,
)


def run_dataset_distribution_analysis(
    source_directory_path,
    output_image_path,
    should_display_figure,
):
    """Collect the images, count them per class and plot the charts."""
    class_distribution = build_class_distribution_of_directory(
        source_directory_path
    )
    print_distribution_summary(class_distribution)
    chart_renderer = build_chart_renderer_for_directory(source_directory_path)
    distribution_figure = chart_renderer.render_distribution_figure(
        class_distribution
    )
    if output_image_path is not None:
        chart_renderer.save_figure_to_file(
            distribution_figure,
            output_image_path,
        )
        print("Chart written to: " + str(output_image_path))
    if should_display_figure:
        display_figure_when_display_is_available(distribution_figure)


def build_class_distribution_of_directory(source_directory_path):
    """Return the class distribution of the images under a directory."""
    image_file_collector = ImageFileCollector(source_directory_path)
    grouped_image_paths = (
        image_file_collector.collect_image_paths_grouped_by_class()
    )
    return ClassDistribution.from_grouped_image_paths(grouped_image_paths)


def build_chart_renderer_for_directory(source_directory_path):
    """Return a renderer titled after the analysed directory name."""
    plant_type_label = extract_plant_type_label_from_directory_path(
        source_directory_path
    )
    chart_title = build_chart_title_from_plant_type_label(plant_type_label)
    return DistributionChartRenderer(chart_title)
