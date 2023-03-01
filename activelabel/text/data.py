import random

from torch.utils.data import Dataset

from activelabel.bases import get_text_files


class TextClassificationDataset(Dataset):
    def __init__(self, directory, labels, label_map):
        self.files = get_text_files(directory)
        random.shuffle(self.files)
        self.labels = labels
        self.label_map = label_map

    def __getitem__(self, index):
        file = self.files[index]
        text = file.read_text()

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
