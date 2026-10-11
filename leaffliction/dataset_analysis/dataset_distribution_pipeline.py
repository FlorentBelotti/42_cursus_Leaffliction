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
from leaffliction.dataset_analysis.plant_type_grouper import (
    group_image_paths_by_plant_type,
)
from leaffliction.dataset_analysis.plant_type_label_extractor import (
    build_chart_title_from_plant_type_label,
)
from leaffliction.utils.figure_display import (
    display_figure_when_display_is_available,
)


def run_dataset_distribution_analysis(
    source_directory_path,
    output_image_path,
    should_display_figure,
):
    """Collect the images, then plot one pie and bar chart per plant."""
    image_file_collector = ImageFileCollector(source_directory_path)
    grouped_image_paths = (
        image_file_collector.collect_image_paths_grouped_by_class()
    )
    grouped_by_plant = group_image_paths_by_plant_type(
        grouped_image_paths,
        source_directory_path,
    )
    last_figure = None
    for plant_label in grouped_by_plant:
        last_figure = _analyse_one_plant(
            plant_label,
            grouped_by_plant[plant_label],
            _build_plant_output_path(
                output_image_path,
                plant_label,
                len(grouped_by_plant) > 1,
            ),
        )
    if should_display_figure:
        display_figure_when_display_is_available(last_figure)


def _analyse_one_plant(plant_label, grouped_image_paths, output_image_path):
    """Print the summary and render the charts of one plant type."""
    class_distribution = ClassDistribution.from_grouped_image_paths(
        grouped_image_paths
    )
    print("[" + plant_label + "]")
    print_distribution_summary(class_distribution)
    chart_renderer = DistributionChartRenderer(
        build_chart_title_from_plant_type_label(plant_label)
    )
    distribution_figure = chart_renderer.render_distribution_figure(
        class_distribution
    )
    if output_image_path is not None:
        chart_renderer.save_figure_to_file(
            distribution_figure,
            output_image_path,
        )
        print("Chart written to: " + str(output_image_path))
    return distribution_figure


def _build_plant_output_path(output_image_path, plant_label, several_plants):
    """Return the output file of one plant, suffixed when there are many."""
    if output_image_path is None or not several_plants:
        return output_image_path
    return output_image_path.with_name(
        output_image_path.stem
        + "_"
        + plant_label
        + output_image_path.suffix
    )
