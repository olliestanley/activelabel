from pathlib import Path
from typing import Any, Tuple

import pandas as pd
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

    def setup(self, source_directory: Path, initial: pd.DataFrame = None) -> None:
        raise NotImplementedError("Implement this method.")

    def next_sample(self) -> Tuple[str, Any, Any, float]:
        raise NotImplementedError("Implement this method.")

    def update_model(self) -> None:
        raise NotImplementedError("Implement this method.")

    def update_predictions(self) -> None:
        raise NotImplementedError("Implement this method.")
