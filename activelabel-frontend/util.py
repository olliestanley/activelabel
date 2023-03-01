from pathlib import Path
from typing import Sequence, Union

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]
TEXT_EXTENSIONS = [".txt"]

PathLike = Union[Path, str]


def get_image_files(directory: Path) -> Sequence[Path]:
    return [
        file for file in directory.iterdir()
        if file.suffix in IMAGE_EXTENSIONS
    ]


def get_text_files(directory: Path) -> Sequence[Path]:
    return [
        file for file in directory.iterdir()
        if file.suffix in TEXT_EXTENSIONS
    ]


def infer_mode(source_directory: Path) -> str:
    num_images = len(get_image_files(source_directory))
    num_texts = len(get_text_files(source_directory))
    return "image" if num_images >= num_texts else "text"
