import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader


class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=1):
        super().__init__()
        self.cv = count_vectorizer
        self.num_classes = None
        self.classes = None
        self.number_of_features = None
        self.number_of_samples = None
        self.prior = None
        self.feature_probabilities = None
        self.log_probs = None
        self.log_feature_probabilities = None
        self.log_prior = None
        self.alpha = alpha

    def fit(self, x, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

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
        self.number_of_samples, self.number_of_features = x.shape
        self.classes = np.unique(y)
        self.num_classes = len(self.classes)
        self.prior = np.zeros(self.num_classes)
        for i, label in enumerate(self.classes):
            self.prior[i] = np.sum(y == label) / self.number_of_samples
        self.feature_probabilities = np.zeros((self.num_classes, self.number_of_features))
        for i, label in enumerate(self.classes):
            labels = x[y == label]
            self.feature_probabilities[i, :] = (np.sum(labels, axis=0) + self.alpha) / (
                        np.sum(labels) + self.alpha * self.number_of_features)
        self.log_feature_probabilities = np.log(self.feature_probabilities)
        self.log_prior = np.log(self.prior)
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
        probs = x @ self.log_feature_probabilities.T + self.log_prior
        return self.classes[np.argmax(probs, axis=1)]

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

    def get_percent_of_positive_reviews(self, sentences):
        """
        You have to override this method because we are using a different embedding method in this class.
        """
        loader = ReviewLoader('../IMDB Dataset.csv')
        loader.load_data()
        positive_index = loader.sentiments.index('positive')
        label_encoder = LabelEncoder()
        loader.sentiments = label_encoder.fit_transform(loader.sentiments)
        return 100 * (np.sum(self.predict(self.cv.transform(sentences).toarray()) == loader.sentiments[positive_index]) / len(sentences))


# F1 Accuracy : 85%
if __name__ == '__main__':
    """
    First, find the embeddings of the revies using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn's CountVectorizer to find the embeddings.
    """
    loader = ReviewLoader('../IMDB Dataset.csv')
    loader.load_data()
    reviews = loader.review_tokens
    sentiments = np.array(loader.sentiments)
    x_train, x_test, y_train, y_test = train_test_split(reviews, sentiments, test_size=0.2)
    cv = CountVectorizer()
    x_train_cv = cv.fit_transform(x_train)
    x_test_cv = cv.transform(x_test)
    nb_classifier = NaiveBayes(cv, alpha=1)
    nb_classifier.fit(x_train_cv, y_train)
    y_pred = nb_classifier.predict(x_test_cv)
    print(classification_report(y_test, y_pred))
