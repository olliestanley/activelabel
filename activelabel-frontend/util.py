from enum import Enum
from pathlib import Path

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]
TEXT_EXTENSIONS = [".txt"]

PathLike = Path | str


class LabelMode(Enum, str):
    IMAGE = "image"
    TEXT = "text"


def count_image_files(directory: Path) -> int:
    return sum(1 for file in directory.iterdir() if file.suffix in IMAGE_EXTENSIONS)


def count_text_files(directory: Path) -> int:
    return sum(1 for file in directory.iterdir() if file.suffix in TEXT_EXTENSIONS)


def infer_mode(source_directory: Path) -> LabelMode:
    num_images = count_image_files(source_directory)
    num_texts = count_image_files(source_directory)
    return LabelMode.IMAGE if num_images >= num_texts else LabelMode.TEXT
