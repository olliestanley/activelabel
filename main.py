import argparse
from pathlib import Path
from typing import Callable

from image import label_images
from text import label_texts
from util import infer_mode


def get_label_function(mode: str) -> Callable:
    if mode == "image":
        return label_images
    if mode == "text":
        return label_texts
    return None


def run(mode: str, labels: str, source: Path, out: Path):
    label_function = get_label_function(mode)
    label_df = label_function(source, labels)
    label_df.to_csv(out, index=False)


def main(args):
    source = Path(args.source)
    mode = infer_mode(source) if args.mode == "infer" else args.mode
    run(mode, args.labels, source, Path(args.out))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="infer")
    parser.add_argument("--labels", type=str, default="class")
    parser.add_argument("--source", type=str, default="data/raw")
    parser.add_argument("--out", type=str, default="data/labels.csv")
    args = parser.parse_args()
    main(args)
