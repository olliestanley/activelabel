import random
from pathlib import Path

from torch.utils.data import Dataset

from activelabel.util import CachedFunctionKV, get_text

TEXT_EXTENSIONS = [".txt"]


def get_text_files(directory: Path) -> list[Path]:
    return [file for file in directory.iterdir() if file.suffix in TEXT_EXTENSIONS]


class TextClassificationDataset(Dataset):
    def __init__(
        self,
        directory: Path,
        labels: dict[str, list],
        label_map,
        use_cache: bool = False,
    ):
        self.files = get_text_files(directory)
        random.shuffle(self.files)
        self.labels = labels
        self.label_map = label_map

        self.getter = CachedFunctionKV(get_text) if use_cache else get_text

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

    def count_unlabelled_samples(self) -> int:
        return len(self.files) - len(self.labels["filename"])
