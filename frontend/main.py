import argparse
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any
import yaml

from activelabel import LabelJob
from activelabel.text.jobs import TextClassificationLabelJob
from activelabel.text.models import Word2VecSVCTextClassifier
from activelabel.util import LabelingError
import polars as pl


class JobStatus(Enum, int):
    OK = 0
    COMPLETE = 1
    EXIT = -1


def perform_labelling(label_job: LabelJob) -> None:
    complete = False

    while True:
        for _ in range(label_job.interval):
            status = label_sample(label_job)

            if status == JobStatus.COMPLETE:
                complete = True
                break

            if status == JobStatus.EXIT:
                return

        print("Updating model...")
        label_job.update_model()

        if not complete:
            print("Updating predictions...")
            label_job.update_predictions()


def label_sample(label_job: LabelJob) -> JobStatus:
    try:
        identifier, sample, pred, conf = label_job.next_sample()
    except LabelingError:
        return JobStatus.COMPLETE

    print(sample)
    print(f"Predicted: {pred} (Confidence: {round(conf, 3)})")

    label = input()
    if label == "exit":
        return JobStatus.EXIT

    label_job.labels["filename"].append(identifier)
    label_job.labels["label"].append(label)
    return JobStatus.OK


def get_job(mode: str, label_type: str, interval: int) -> LabelJob:
    if mode == "text":
        if label_type == "class":
            model = Word2VecSVCTextClassifier(["+", "-", "="])
            return TextClassificationLabelJob(model, interval=interval)

    raise NotImplementedError(f"Unavailable mode / label type combination: {mode} / {label_type}")


@dataclass
class Config:
    mode: str
    label_type: str
    source: Path
    out: Path
    initial: Path
    interval: int

    def __post_init__(self):
        self.source = Path(self.source)
        self.out = Path(self.out)
        self.initial = Path(self.initial)
        self.interval = int(self.interval)

        self.source.mkdir(parents=True, exist_ok=True)


def main(config_dict: dict[str, Any]) -> None:
    config = Config(**config_dict)

    initial_df = pl.read_csv(config.initial) if config.initial.exists() else None

    label_job = get_job(config.mode, config.label_type, config.interval)
    label_job.setup(config.source, initial_df)

    perform_labelling(label_job)
    # TODO: save model

    label_df = pl.from_dict(label_job.labels)
    label_df.write_csv(config.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="frontend/config.yml")
    args = parser.parse_args()

    config_file = Path(args.config)
    config = yaml.safe_load(config_file.read_text())

    main(config)
