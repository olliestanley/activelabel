from pathlib import Path
from typing import Dict, Sequence

import pandas as pd
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]
TEXT_EXTENSIONS = [".txt"]


class ModelWrapper:
    def fit(self, data: Dataset):
        raise NotImplementedError("Implement this method.")

    def predict_with_confidence(self, input) -> tuple:
        raise NotImplementedError("Implement this method.")


class ClassifierWrapper(ModelWrapper):
    def __init__(self, classes):
        super().__init__()

        self.classes = classes
        self.class_map = {cl: i for i, cl in enumerate(classes)}


class LabelJob:
    def __init__(self, model: ModelWrapper, interval: int = 50):
        self.model = model
        self.interval = interval

    def run(
        self, source_directory: Path, initial: pd.DataFrame = None
    ) -> pd.DataFrame:
        raise NotImplementedError("Implement this method.")


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
