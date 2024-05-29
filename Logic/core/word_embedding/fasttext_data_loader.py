import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from nltk.corpus import stopwords
import string
from nltk.tokenize import word_tokenize


def preprocess_text(text, minimum_length=1, stopword_removal=True, stopwords_domain=[], lower_case=True,
                       punctuation_removal=True):
    """
    preprocess text by removing stopwords, punctuations, and converting to lowercase, and also filter based on a min length
    for stopwords use nltk.corpus.stopwords.words('english')
    for punctuations use string.punctuation

    Parameters
    ----------
    text: str
        text to be preprocessed
    minimum_length: int
        minimum length of the token
    stopword_removal: bool
        whether to remove stopwords
    stopwords_domain: list
        list of stopwords to be removed base on domain
    lower_case: bool
        whether to convert to lowercase
    punctuation_removal: bool
        whether to remove punctuations
    """
    if lower_case:
        text = text.lower()
    tokens = word_tokenize(text)
    if stopword_removal:
        stopword = set(stopwords_domain) | set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stopword]
    if punctuation_removal:
        tokens = [word for word in tokens if word not in string.punctuation]
    if minimum_length > 1:
        tokens = [word for word in tokens if len(word) >= minimum_length]
    return ' '.join(tokens)


def row_to_text(row):
    if not len(row['reviews']):
        return None
    synopses_text = ' '.join(row['synposis'])
    summaries_text = ' '.join(row['summaries'])
    reviews_text = ' '.join(row['reviews'][0])
    title_text = row['title']
    return ' '.join([synopses_text, summaries_text, reviews_text, title_text])


class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synopses, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """

    def __init__(self, file_path):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.file_path = file_path

    def read_data_to_df(self):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synopses, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synopses, summaries, reviews, titles, genres)
        """
        return pd.read_json(self.file_path)[["synposis", "summaries", "reviews", "title", "genres"]]

    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        df = self.read_data_to_df().dropna()
        X = df.apply(row_to_text, axis=1).dropna().apply(preprocess_text).values
        y = LabelEncoder().fit_transform(df['genres'].apply(lambda x: np.random.choice(x)).values)
        return X, y
