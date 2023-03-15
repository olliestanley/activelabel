from pathlib import Path
from typing import Any, Tuple

import numpy as np
import polars as pl

from activelabel.bases import LabelJob
from activelabel.text.data import TextClassificationDataset
from activelabel.text.models import ClassifierWrapper
from activelabel.util import LabelingError


class TextClassificationLabelJob(LabelJob):
    def __init__(self, model: ClassifierWrapper, interval: int = 50):
        super().__init__(model, interval)

        self.preds = []
        self.confs = []

    def setup(self, source_directory: Path, initial: pl.DataFrame = None) -> None:
        if initial is None:
            self.labels = {
                "filename": [],
                "label": [],
            }
        else:
            self.labels = initial.to_dict(as_series=False)

        self.dataset = TextClassificationDataset(
            source_directory, self.labels, self.model.class_map
        )

        if len(self.dataset) == 0:
            raise LabelingError("No data to label")

        self.confs = [0 for _ in range(len(self.dataset))]
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

    def update_model(self):
        self.model.fit(self.dataset)

    def update_predictions(self):
        preds, confs = [], []

        for x, _ in self.dataset:
            pred, conf = self.model.predict_with_confidence(x)
            preds.append(pred)
            confs.append(conf)

        self.preds, self.confs = preds, confs
