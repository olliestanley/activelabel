import re

import gensim.downloader as gensim_api
import nltk
import numpy as np
from sklearn import svm
from torch.utils.data import Dataset

from activelabel.util import ModelWrapper


class ClassifierWrapper(ModelWrapper):
    def __init__(self, classes):
        super().__init__()

        self.classes = classes
        self.class_map = {cl: i for i, cl in enumerate(classes)}


class Word2VecSVCTextClassifier(ClassifierWrapper):
    def __init__(self, classes, max_length: int = 500):
        super().__init__(classes)

        nltk.download("stopwords")
        nltk.download("wordnet")
        nltk.download("omw-1.4")

        self.stopwords = nltk.corpus.stopwords.words("english")
        self.stemmer = nltk.stem.PorterStemmer()
        self.lemmatizer = nltk.stem.WordNetLemmatizer()
        self.word2vec = gensim_api.load("word2vec-google-news-300")

        self.model = svm.SVC(probability=True)
        self.max_length = max_length

    def preprocess_text(self, text: str) -> np.ndarray:
        text = re.sub(r'[^\w\s]', '', str(text).lower().strip())
        lst_text = text.split()
        lst_text = [word for word in lst_text if word not in self.stopwords]
        lst_text = [self.stemmer.stem(word) for word in lst_text]
        lst_text = [self.lemmatizer.lemmatize(word) for word in lst_text]
        arr = np.concatenate([self.word2vec[word] for word in lst_text])
        arr = np.pad(arr, (self.max_length - len(arr), 0), "constant")
        return arr

    def fit(self, data: Dataset):
        datapoints = [data[i] for i in range(len(data))]
        train_x = [
            self.preprocess_text(val[0])
            for val in datapoints if val[1] is not None
        ]
        train_y = [val[1] for val in datapoints if val[1] is not None]

        self.model = svm.SVC(probability=True)
        self.model.fit(train_x, train_y)

    def predict_with_confidence(self, input) -> tuple:
        preprocessed = self.preprocess_text(input)
        pred = self.model.predict([preprocessed])[0]
        pred_class = self.classes[pred]
        prob = self.model.predict_proba([preprocessed])[0][pred]
        return pred_class, prob
