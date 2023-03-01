import argparse
from pathlib import Path

from activelabel import JobManager, LabelJob
import pandas as pd

import util
from util import PathLike


def perform_labelling(label_job: LabelJob):
    while True:
        for _ in range(label_job.interval):
            status = label_sample(label_job)
            if status < 0:
                return

        print("Updating model and predictions...")
        label_job.update_model()
        label_job.update_predictions()


def label_sample(label_job: LabelJob):
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


def main(
    mode: str = "infer",
    label_type: str = "class",
    source: PathLike = "data/raw",
    out: PathLike = "data/output_labels.csv",
    initial: PathLike = "data/input_labels.csv",
    interval: int = 10,
):
    source, out, initial = Path(source), Path(out), Path(initial)
    mode = util.infer_mode(source) if mode == "infer" else mode

    initial_df = pd.read_csv(initial) if initial.exists() else None

    job_manager = JobManager(mode, label_type)
    label_job = job_manager.get_job(interval=interval)
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
