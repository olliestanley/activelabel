import random
from pathlib import Path

import polars as pl

from activelabel import ClassificationDataset
from activelabel.util import CachedFunctionKV, get_text

TEXT_EXTENSIONS = [".txt"]


def get_text_files(directory: Path) -> list[Path]:
    return [file for file in directory.iterdir() if file.suffix in TEXT_EXTENSIONS]


class TextClassificationDataset(ClassificationDataset):
    def __init__(
        self,
        directory: Path,
        labels: dict[str, list],
        label_map: dict[str, int],
        use_cache: bool = False,
    ):
        self.files = get_text_files(directory)
        random.shuffle(self.files)
        self.labels = labels
        self.label_map = label_map

        self.getter = CachedFunctionKV(get_text) if use_cache else get_text

    def __getitem__(self, index: int):
        file = self.files[index]
        text = self.getter(file)

        try:
            label_index = self.labels["filename"].index(file)
            return text, self.label_map[self.labels["label"][label_index]]
        except ValueError:
            return text, None

    def __len__(self):
        return len(self.files)

    def has_label(self, index: int) -> bool:
        try:
            file = self.files[index]
            self.labels["filename"].index(file)
            return True
        except ValueError:
            return False

    def count_unlabelled_samples(self) -> int:
        return len(self.files) - len(self.labels["filename"])

    @classmethod
    def from_files(
        cls,
        source_directory: Path,
        label_map: dict[str, int],
        initial_labels: pl.DataFrame | None = None,
    ) -> ClassificationDataset:
        """
        Build a TextClassificationDataset from a directory containing text files, one per sample.

        Args:
            source_directory (Path): The directory containing the text files.
            label_map (dict[str, int]): A mapping from class names to integer labels.
            initial_labels (pl.DataFrame, optional): A DataFrame containing existing labels.

        Returns:
            A TextClassificationDataset for the text data.
        """

        if initial_labels is None:
            labels = {
                "filename": [],
                "label": [],
            }
        else:
            labels = initial_labels.to_dict(as_series=False)

        dataset = cls(source_directory, labels, label_map)
        return dataset
