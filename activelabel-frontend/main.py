import argparse
from pathlib import Path

from activelabel import LabelJob
from activelabel.text.jobs import TextClassificationLabelJob
from activelabel.text.models import Word2VecSVCTextClassifier
import pandas as pd

from util import PathLike, infer_mode


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


def main(
    mode: str = "infer",
    label_type: str = "class",
    source: PathLike = "data/raw",
    out: PathLike = "data/output_labels.csv",
    initial: PathLike = "data/input_labels.csv",
    interval: int = 10,
) -> None:
    source, out, initial = Path(source), Path(out), Path(initial)
    mode = infer_mode(source) if mode == "infer" else mode

    initial_df = pd.read_csv(initial) if initial.exists() else None

    label_job = get_job(mode, label_type, interval)
    label_job.setup(source, initial_df)

    perform_labelling(label_job)

    label_df = pd.DataFrame.from_dict(label_job.labels)
    label_df.to_csv(out, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="infer")
    parser.add_argument("--label-type", type=str, default="class")
    parser.add_argument("--source", type=str, default="data/raw")
    parser.add_argument("--out", type=str, default="data/output_labels.csv")
    parser.add_argument("--initial", type=str, default="data/input_labels.csv")
    parser.add_argument("--interval", type=int, default=10)
    args = parser.parse_args()

    main(args.mode, args.label_type, args.source, args.out, args.initial, args.interval)
