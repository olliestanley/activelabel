from typing import Tuple

from torch.utils.data import Dataset

from activelabel.bases import ClassifierWrapper


class CNNTextClassifier(ClassifierWrapper):
    def __init__(self, classes, image_shape: Tuple[int, int, int]):
        super().__init__(classes)

        self.image_shape = image_shape
        # TODO

    def fit(self, data: Dataset):
        # TODO
        pass

    def predict_with_confidence(self, input) -> tuple:
        # TODO
        pass
