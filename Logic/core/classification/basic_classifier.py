import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from Logic.core.classification.data_loader import ReviewLoader
from Logic.core.word_embedding.fasttext_model import FastText


class BasicClassifier:
    def __init__(self):
        pass

    def fit(self, x, y):
        raise NotImplementedError()

    def predict(self, x):
        raise NotImplementedError()

    def prediction_report(self, x, y):
        raise NotImplementedError()

    def get_percent_of_positive_reviews(self, sentences):
        """
        Get the percentage of positive reviews in the given sentences
        Parameters
        ----------
        sentences: list
            The list of sentences to get the percentage of positive reviews
        Returns
        -------
        float
            The percentage of positive reviews
        """
        loader = ReviewLoader('../IMDB Dataset.csv')
        loader.load_data()
        loader.review_tokens = sentences
        loader.get_embeddings()
        positive_index = loader.sentiments.index('positive')
        label_encoder = LabelEncoder()
        loader.sentiments = label_encoder.fit_transform(loader.sentiments)
        return 100 * (np.sum(self.predict(loader.embeddings) == loader.sentiments[positive_index]) / len(sentences))
