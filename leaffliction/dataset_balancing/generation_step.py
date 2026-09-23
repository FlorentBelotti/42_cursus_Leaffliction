"""One unit of work of the balancing stage."""

from dataclasses import dataclass
from pathlib import Path

from leaffliction.image_augmentation.base_augmentation import BaseAugmentation


@dataclass
class GenerationStep:
    """Describe which image to augment, how, and under which variant."""

    source_image_path: Path
    augmentation: BaseAugmentation
    variation_index: int
