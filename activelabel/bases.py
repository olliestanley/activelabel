from typing import Any

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


class ClassificationDataset(Dataset):
    def has_label(self, index: int) -> bool:
        raise NotImplementedError("Implement this method.")

    def count_unlabelled_samples(self) -> int:
        raise NotImplementedError("Implement this method.")


class LabelJob:
    def __init__(self, model: ModelWrapper, dataset: Dataset, interval: int = 50):
        self.model = model
        self.dataset = dataset
        self.interval = interval

    def next_sample(self) -> tuple[str, Any, Any, float]:
        """Raises LabelingError if no more samples to label."""

        raise NotImplementedError("Implement this method.")

    def add_label(self, identifier: str, label: Any) -> None:
        raise NotImplementedError("Implement this method.")

    def update_model(self) -> None:
        raise NotImplementedError("Implement this method.")

    def update_predictions(self) -> None:
        raise NotImplementedError("Implement this method.")
