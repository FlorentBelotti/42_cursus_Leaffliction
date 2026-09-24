"""Decide how many images every class must gain to be balanced."""

from leaffliction.dataset_balancing.class_balancing_plan import (
    ClassBalancingPlan,
)


class BalancingPlanBuilder:
    """Turn a class distribution into one balancing plan per class.

    The target size is the size of the most populated class: only new
    images are created, no original image is ever discarded. Under
    sampling the majority class would have been the alternative, but it
    throws away real observations, which is the wrong trade when the
    smallest classes already hold few hundred images.
    """

    def build_plans_for_distribution(self, class_distribution):
        """Return one plan per class, in alphabetical class order."""
        target_image_count = self._compute_target_image_count(
            class_distribution
        )
        balancing_plans = []
        class_names = (
            class_distribution.get_class_names_sorted_alphabetically()
        )
        for class_name in class_names:
            balancing_plans.append(
                self._build_plan_for_class(
                    class_name,
                    class_distribution,
                    target_image_count,
                )
            )
        return balancing_plans

    def _compute_target_image_count(self, class_distribution):
        """Return the size every class has to reach after balancing."""
        return class_distribution.get_largest_class_image_count()

    def _build_plan_for_class(
        self,
        class_name,
        class_distribution,
        target_image_count,
    ):
        """Return the plan bringing one class to the target size."""
        existing_image_count = class_distribution.get_image_count_for_class(
            class_name
        )
        return ClassBalancingPlan(
            class_name=class_name,
            existing_image_count=existing_image_count,
            images_to_generate_count=(
                target_image_count - existing_image_count
            ),
        )
