import argparse
from pathlib import Path
from typing import Any
import yaml

from activelabel import LabelJob
from activelabel.text.jobs import TextClassificationLabelJob
from activelabel.text.models import Word2VecSVCTextClassifier
import polars as pl


class Config:
    def __init__(self, config: dict[str, Any]):
        self.mode = config["Mode"]
        self.label_type = config["Label-Type"]
        self.source = Path(config["Data-Directory"])
        self.out = Path(config["Input-Labels"])
        self.initial = Path(config["Output-Labels"])
        self.interval = int(config["Retrain-Interval"])


def perform_labelling(label_job: LabelJob) -> None:
    while True:
        for _ in range(label_job.interval):
            status = label_sample(label_job)
            if status < 0:
                return

        print("Updating model and predictions...")
        label_job.update_model()
        label_job.update_predictions()


def label_sample(label_job: LabelJob) -> int:
    identifier, sample, pred, conf = label_job.next_sample()
    if sample is None:
        return -1

    print(sample)
    print(f"Predicted: {pred} (Confidence: {round(conf, 3)})")

    label = input()
    if label == "exit":
        return -2

    label_job.labels["filename"].append(identifier)
    label_job.labels["label"].append(label)
    return 0


def get_job(mode: str, label_type: str, interval: int) -> LabelJob:
    if mode == "text":
        if label_type == "class":
            model = Word2VecSVCTextClassifier(["+", "-", "="])
            return TextClassificationLabelJob(model, interval=interval)

    raise NotImplementedError(f"Unavailable mode / label type combination: {mode} / {label_type}")


def main(config_dict: dict[str, Any]) -> None:
    config = Config(config_dict)

    initial_df = pl.read_csv(config.initial) if config.initial.exists() else None

    label_job = get_job(config.mode, config.label_type, config.interval)
    label_job.setup(config.source, initial_df)

    perform_labelling(label_job)

    label_df = pl.from_dict(label_job.labels)
    label_df.write_csv(config.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="frontend/config.yml")
    args = parser.parse_args()

    config_file = Path(args.config)
    config = yaml.safe_load(config_file.read_text())

    main(config)
