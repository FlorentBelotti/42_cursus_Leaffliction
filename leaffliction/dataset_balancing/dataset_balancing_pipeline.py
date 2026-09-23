"""Chronological orchestration of the whole data set balancing."""

from leaffliction.dataset_analysis.class_distribution import ClassDistribution
from leaffliction.dataset_analysis.distribution_summary_printer import (
    print_distribution_summary,
)
from leaffliction.dataset_analysis.image_file_collector import (
    ImageFileCollector,
)
from leaffliction.dataset_balancing.augmented_dataset_copier import (
    AugmentedDatasetCopier,
)
from leaffliction.dataset_balancing.balancing_plan_builder import (
    BalancingPlanBuilder,
)
from leaffliction.dataset_balancing.balancing_report_printer import (
    print_balancing_report,
)
from leaffliction.dataset_balancing.dataset_balancer import DatasetBalancer
from leaffliction.image_augmentation.augmentation_registry import (
    AugmentationRegistry,
)


def run_dataset_balancing(
    source_directory_path,
    destination_directory_path,
    should_overwrite_destination=False,
):
    """Copy the data set, then augment the copy until it is balanced."""
    augmented_dataset_copier = AugmentedDatasetCopier()
    augmented_dataset_copier.raise_when_destination_cannot_be_used(
        destination_directory_path,
        should_overwrite_destination,
    )
    grouped_image_paths = _collect_grouped_image_paths(source_directory_path)
    class_distribution = ClassDistribution.from_grouped_image_paths(
        grouped_image_paths
    )
    print_distribution_summary(class_distribution)
    balancing_plans = BalancingPlanBuilder().build_plans_for_distribution(
        class_distribution
    )
    augmented_dataset_copier.copy_source_dataset_to_destination(
        source_directory_path,
        destination_directory_path,
    )
    _generate_every_missing_image(
        balancing_plans,
        grouped_image_paths,
        source_directory_path,
        destination_directory_path,
    )
    print_balancing_report(balancing_plans, destination_directory_path)


def _collect_grouped_image_paths(source_directory_path):
    """Return the source images of every class, in a stable order."""
    image_file_collector = ImageFileCollector(source_directory_path)
    return image_file_collector.collect_image_paths_grouped_by_class()


def _generate_every_missing_image(
    balancing_plans,
    grouped_image_paths,
    source_directory_path,
    destination_directory_path,
):
    """Run the balancer on every class that is short of images."""
    dataset_balancer = DatasetBalancer(AugmentationRegistry())
    for balancing_plan in balancing_plans:
        if balancing_plan.is_already_balanced():
            continue
        class_image_paths = grouped_image_paths[balancing_plan.class_name]
        dataset_balancer.generate_missing_images_for_class(
            balancing_plan,
            class_image_paths,
            _resolve_destination_class_directory(
                class_image_paths[0].parent,
                source_directory_path,
                destination_directory_path,
            ),
        )


def _resolve_destination_class_directory(
    source_class_directory_path,
    source_directory_path,
    destination_directory_path,
):
    """Return the copy of a class directory inside the destination."""
    relative_class_directory_path = source_class_directory_path.relative_to(
        source_directory_path
    )
    return destination_directory_path / relative_class_directory_path
