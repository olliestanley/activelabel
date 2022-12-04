import argparse
from pathlib import Path

import pandas as pd

from activelabel.text.classify import TextClassificationLabelJob, TextClassifier
from activelabel.util import LabelJob, ModelWrapper, infer_mode


def get_model(mode: str, label_type: str) -> ModelWrapper:
    # TODO
    return TextClassifier(["+", "-", "="])


def get_label_job(mode: str, label_type: str) -> LabelJob:
    # TODO
    model = get_model(mode, label_type)
    return TextClassificationLabelJob(model, interval=10)


def run(mode: str, label_type: str, source: Path, out: Path, initial: Path):
    initial_df = pd.read_csv(initial) if initial.exists() else None
    label_job = get_label_job(mode, label_type)
    label_df = label_job.start(source, initial_df)
    label_df.to_csv(out, index=False)


def main(args):
    source = Path(args.source)
    mode = infer_mode(source) if args.mode == "infer" else args.mode
    initial_path = Path(args.initial)
    run(mode, args.label_type, source, Path(args.out), initial_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="infer")
    parser.add_argument("--label-type", type=str, default="class")
    parser.add_argument("--source", type=str, default="data/raw")
    parser.add_argument("--out", type=str, default="data/output_labels.csv")
    parser.add_argument("--initial", type=str, default="data/input_labels.csv")
    args = parser.parse_args()
    main(args)
