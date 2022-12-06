from pathlib import Path

import numpy as np
import pandas as pd

from activelabel.text.data import TextClassificationDataset
from activelabel.text.models import ClassifierWrapper
from activelabel.util import LabelJob


class TextClassificationLabelJob(LabelJob):
    def __init__(self, model: ClassifierWrapper, interval: int = 50):
        super().__init__(model, interval)

        self.preds = []
        self.confs = []

    def run(
        self, source_directory: Path, initial: pd.DataFrame = None
    ) -> pd.DataFrame:
        if initial is None:
            self.labels = {
                "filename": [],
                "label": [],
            }
        else:
            self.labels = initial.to_dict("list")

        self.dataset = TextClassificationDataset(
            source_directory, self.labels, self.model.class_map
        )

        self.confs = [0 for _ in range(len(self.dataset))]
        self.preds = [self.model.classes[c] for c in self.confs]

        self.perform_labelling()

        return pd.DataFrame.from_dict(self.labels)

    def perform_labelling(self):
        while True:
            for _ in range(self.interval):
                status = self.label_sample()
                if status < 0:
                    return

            self.model.fit(self.dataset)
            self.update_predictions()

    def label_sample(self):
        if not self.dataset.has_unlabelled_samples():
            return -1

        index, pred, conf = self.get_lowest_confidence()
        text, _ = self.dataset[index]

        print(text)
        print(f"Predicted: {pred} (Confidence: {round(conf, 3)})")

        label = input()

        if label == "exit":
            return -2

        self.labels["filename"].append(self.dataset.files[index])
        self.labels["label"].append(label)
        return 0

    def update_predictions(self):
        preds, confs = [], []

        for x, _ in self.dataset:
            pred, conf = self.model.predict_with_confidence(x)
            preds.append(pred)
            confs.append(conf)

        self.preds, self.confs = preds, confs

    def get_lowest_confidence(self):
        confs = self.confs.copy()

        index = np.argmin(confs)
        while self.dataset.has_label(index):
            confs[index] = 10
            index = np.argmin(confs)

        return index, self.preds[index], confs[index]
