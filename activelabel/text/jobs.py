from typing import Any, Tuple

import numpy as np
import polars as pl

from activelabel import ClassificationDataset, ClassifierWrapper, LabelJob
from activelabel.util import LabelingError


class TextClassificationLabelJob(LabelJob):
    def __init__(
        self,
        model: ClassifierWrapper,
        dataset: ClassificationDataset,
        interval: int = 50,
    ):
        super().__init__(model, dataset, interval)

        if len(dataset) == 0:
            raise LabelingError("No data to label")

        self.confs = [0 for _ in range(len(dataset))]
        self.preds = [self.model.classes[c] for c in self.confs]

    def next_sample(self) -> Tuple[str, Any, Any, float]:
        if self.dataset.count_unlabelled_samples() < 1:
            raise LabelingError("No more samples to label")

        confs = self.confs.copy()

        index = np.argmin(confs)
        while self.dataset.has_label(index):
            confs[index] = 10
            index = np.argmin(confs)

        return self.dataset.files[index], index, self.preds[index], confs[index]

    def add_label(self, identifier: str, label: Any) -> None:
        self.dataset.labels["filename"].append(identifier)
        self.dataset.labels["label"].append(label)

    def update_model(self) -> None:
        self.model.fit(self.dataset)

    def update_predictions(self) -> None:
        preds, confs = [], []

        for x, _ in self.dataset:
            pred, conf = self.model.predict_with_confidence(x)
            preds.append(pred)
            confs.append(conf)

        self.preds, self.confs = preds, confs

    def get_labels(self) -> pl.DataFrame:
        return pl.from_dict(self.dataset.labels)
