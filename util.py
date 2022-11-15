from pathlib import Path
from typing import Dict, Sequence


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]
TEXT_EXTENSIONS = [".txt"]


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


def get_user_command(key_command_mapping: Dict[str, str]) -> str:
    user_input = input()

    try:
        return key_command_mapping[user_input]
    except KeyError:
        print(f"No command associated with: {user_input}")
        return get_user_command(key_command_mapping)
