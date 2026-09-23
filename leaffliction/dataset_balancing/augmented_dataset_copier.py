"""Copy the original data set before augmenting the copy.

The subject asks for the new images to sit in the class directories and
for the whole thing to be handed in as an augmented directory. Working
on a copy keeps the original data set untouched, which matters because
its sha1 signature is what the evaluation compares.
"""

import shutil

from leaffliction.utils.error_reporting import DestinationAlreadyExistsError


class AugmentedDatasetCopier:
    """Duplicate a data set tree into the destination directory."""

    def copy_source_dataset_to_destination(
        self,
        source_directory_path,
        destination_directory_path,
    ):
        """Copy the whole source tree, then return the destination."""
        shutil.copytree(
            source_directory_path,
            destination_directory_path,
            dirs_exist_ok=True,
        )
        return destination_directory_path

    def raise_when_destination_cannot_be_used(
        self,
        destination_directory_path,
        should_overwrite_destination,
    ):
        """Refuse to write into an existing directory by accident.

        This check is deliberately callable on its own, so that the
        pipeline can reject a bad destination before spending minutes
        walking the source data set.
        """
        if not destination_directory_path.exists():
            return
        if should_overwrite_destination:
            return
        raise DestinationAlreadyExistsError(
            "destination already exists, remove it or pass --force: "
            + str(destination_directory_path)
        )
