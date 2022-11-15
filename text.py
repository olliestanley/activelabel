import re
from pathlib import Path
from typing import Sequence

import gensim.downloader as gensim_api
import nltk
import numpy as np
import pandas as pd
from sklearn import svm

from util import get_text_files, get_user_command

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")


STOPWORDS = nltk.corpus.stopwords.words("english")
STEMMER = nltk.stem.PorterStemmer()
LEMMATIZER = nltk.stem.WordNetLemmatizer()


def get_text_classification_model():
    return svm.SVC()


def preprocess_text(text: str) -> Sequence[str]:
    text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
    lst_text = text.split()
    lst_text = [word for word in lst_text if word not in STOPWORDS]
    lst_text = [STEMMER.stem(word) for word in lst_text]
    lst_text = [LEMMATIZER.lemmatize(word) for word in lst_text]
    return lst_text


def vectorize_text(text: Sequence[str], word2vec, max_length: int = 1500):
    arr = np.concatenate([word2vec[word] for word in text])
    arr = np.pad(arr, (max_length - len(arr), 0), "constant")
    return arr


def predict_text(vectorized_text, model):
    return model.predict([vectorized_text])[0]


def label_texts(source_directory: Path, label_type: str) -> pd.DataFrame:
    if label_type != "class":
        raise ValueError("Text labels only support classification")

    print("Enter maximum number of words per text (integer):")
    max_length = int(input())

    print("Enter space-separated list of categories:")
    categories = input().split(" ")
    cat2id = {cat: idx for idx, cat in enumerate(categories)}

    key_category_mapping = {}
    for category in categories:
        print(f"Enter key corresponding to: {category}")
        key_category_mapping[input()] = category

    label_df = pd.DataFrame(columns=["filename", "label"])

    word2vec = gensim_api.load("word2vec-google-news-300")

    model, is_fit, interval = get_text_classification_model(), False, 3
    train_X, train_y = [], []

    for i, file in enumerate(get_text_files(source_directory)):
        text = file.read_text()

        processed_text = preprocess_text(text)
        vectorized_text = vectorize_text(processed_text, word2vec, max_length)

        if i > 0 and i % interval == 0:
            model = model.fit(train_X, train_y)
            is_fit = True

        print(text)

        if is_fit:
            predicted_category = categories[predict_text(vectorized_text, model)]
            print(f"Suggested: {predicted_category}")

        file_category = get_user_command(key_category_mapping)

        train_X.append(vectorized_text)
        train_y.append(cat2id[file_category])

        label = pd.DataFrame(
            [[str(file), file_category]],
            columns=["filename", "label"]
        )

        label_df = pd.concat([label_df, label], ignore_index=True)
        label_df.to_csv("data/temp/checkpoint.csv", index=False)

    return label_df
