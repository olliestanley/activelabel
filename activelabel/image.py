from pathlib import Path

import pandas as pd

from activelabel.util import LabelJob, get_image_files, get_user_command


class ImageLabelJob(LabelJob):
    def __init__(self):
        super().__init__()

    def get_image_classification_model():
        pass

    def predict_image(image, model):
        pass

    def start(self, source_directory: Path, label_type: str) -> pd.DataFrame:
        if label_type != "class":
            raise ValueError("Image labels only support classification")

        print("Enter space-separated list of categories:")
        categories = input().split(" ")
        cat2id = {cat: idx for idx, cat in enumerate(categories)}

        key_category_mapping = {}
        for category in categories:
            print(f"Enter key corresponding to: {category}")
            key_category_mapping[input()] = category

        label_df = pd.DataFrame(columns=["filename", "label"])

        model, is_fit, interval = self.get_image_classification_model(), False, 3
        train_X, train_y = [], []

        for i, file in enumerate(get_image_files(source_directory)):
            # TODO
            image = None

            if i > 0 and i % interval == 0:
                # TODO
                model = None
                is_fit = True

            # TODO: display image

            if is_fit:
                predicted_category = categories[self.predict_image(image, model)]
                print(f"Suggested: {predicted_category}")

            file_category = get_user_command(key_category_mapping)

            train_X.append(image)
            train_y.append(cat2id[file_category])

            label = pd.DataFrame(
                [[str(file), file_category]],
                columns=["filename", "label"]
            )

            label_df = pd.concat([label_df, label], ignore_index=True)
            label_df.to_csv("data/temp/checkpoint.csv", index=False)

        return label_df
