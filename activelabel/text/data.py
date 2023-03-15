from pathlib import Path
import random

from torch.utils.data import Dataset

from activelabel.util import CachedFunctionKV

TEXT_EXTENSIONS = [".txt"]


def get_text_files(directory: Path) -> list[Path]:
    return [
        file for file in directory.iterdir()
        if file.suffix in TEXT_EXTENSIONS
    ]


class TextClassificationDataset(Dataset):
    def __init__(self, directory: Path, labels: dict[str, list], label_map, use_cache: bool = False):
        self.files = get_text_files(directory)
        random.shuffle(self.files)
        self.labels = labels
        self.label_map = label_map

        text_reader = lambda file: file.read_text()
        self.getter = CachedFunctionKV(text_reader) if use_cache else text_reader

    def __getitem__(self, index: int):
        file = self.files[index]
        text = self.getter(file)

        try:
            label_index = self.labels["filename"].index(file)
            return text, self.label_map[self.labels["label"][label_index]]
        except ValueError:
            return text, None

    def __len__(self):
        return len(self.files)

    def has_label(self, index) -> bool:
        try:
            file = self.files[index]
            self.labels["filename"].index(file)
            return True
        except ValueError:
            return False

    def has_unlabelled_samples(self):
        return len(self.files) > len(self.labels["filename"])
