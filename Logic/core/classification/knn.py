import heapq
from collections import Counter

import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader


class KnnClassifier(BasicClassifier):
    def __init__(self, n_neighbors):
        super().__init__()
        self.k = n_neighbors
        self.x_train = None
        self.y_train = None

    def fit(self, x, y):
        """
        Fit the model using X as training data and y as target values
        use the Euclidean distance to find the k nearest neighbors
        Warning: Maybe you need to reduce the size of X to avoid memory errors

        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        self
            Returns self as a classifier
        """
        self.x_train, self.y_train = x, y
        return self

    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        predictions = []
        for test_data in tqdm(x):
            heap = []
            for i, train_data in enumerate(self.x_train):
                distance = np.linalg.norm(test_data - train_data)
                if len(heap) < self.k:
                    heapq.heappush(heap, (-distance, self.y_train[i]))
                else:
                    heapq.heappushpop(heap, (-distance, self.y_train[i]))
            labels = [label for _, label in heap]
            counter = Counter(labels)
            predictions.append(sorted(counter.items(), key=lambda x: x[1])[-1][0])
        return np.array(predictions)

    def prediction_report(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        return classification_report(y, self.predict(x))


# F1 Accuracy : 70%
if __name__ == '__main__':
    """
    Fit the model with the training data and predict the test data, then print the classification report
    """
    loader = ReviewLoader('../IMDB Dataset.csv')
    loader.load_data()
    loader.embeddings = np.load('embeddings.npy')

    X_train, X_test, y_train, y_test = loader.split_data()

    knn = KnnClassifier(5)
    knn.fit(X_train, y_train)
    print(knn.prediction_report(X_test, y_test))
