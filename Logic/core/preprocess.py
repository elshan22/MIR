import re
import string

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

class Preprocessor:

    def __init__(self, documents: list):
        self.documents = documents
        self.stopwords = list(map(lambda x: x.replace('\n', ''), open('stopwords.txt', 'r').readlines()))

    def preprocess(self):
        for i, doc in enumerate(self.documents):
            doc = doc.lower()
            doc = self.remove_links(doc)
            doc = self.remove_punctuations(doc)
            doc = self.remove_stopwords(doc)
            doc = self.normalize(doc)
            self.documents[i] = doc
        return self.documents

    def normalize(self, text: str):
        ps = PorterStemmer()
        return ' '.join(list(map(lambda x: ps.stem(x), self.tokenize(text))))

    def remove_links(self, text: str):
        patterns = [r'\S*http\S*', r'\S*www\S*', r'\S+\.ir\S*', r'\S+\.com\S*', r'\S+\.org\S*', r'\S*@\S*']
        result = []
        for word in self.tokenize(text):
            flag = True
            for p in patterns:
                if re.match(p, word):
                    flag = False
                    break
            if flag: result.append(word)
        return ' '.join(result)

    def remove_punctuations(self, text: str):
        return text.translate(str.maketrans('', '', string.punctuation))

    def tokenize(self, text: str):
        return word_tokenize(text)

    def remove_stopwords(self, text: str):
        return ' '.join(['' if word in self.stopwords else word for word in self.tokenize(text)])
