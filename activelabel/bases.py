from pathlib import Path
from typing import Any, Tuple

import polars as pl
from torch.utils.data import Dataset


class ModelWrapper:
    def fit(self, data: Dataset):
        raise NotImplementedError("Implement this method.")

    def predict_with_confidence(self, input) -> tuple:
        raise NotImplementedError("Implement this method.")


class ClassifierWrapper(ModelWrapper):
    def __init__(self, classes):
        super().__init__()

        self.classes = classes
        self.class_map = {cl: i for i, cl in enumerate(classes)}


class LabelJob:
    def __init__(self, model: ModelWrapper, interval: int = 50):
        self.model = model
        self.interval = interval

    def setup(self, source_directory: Path, initial: pl.DataFrame = None) -> None:
        """Raises LabelingError if no data to label."""

        raise NotImplementedError("Implement this method.")

    def next_sample(self) -> Tuple[str, Any, Any, float]:
        """Raises LabelingError if no more samples to label."""

        raise NotImplementedError("Implement this method.")

    def update_model(self) -> None:
        raise NotImplementedError("Implement this method.")

    def update_predictions(self) -> None:
        raise NotImplementedError("Implement this method.")
