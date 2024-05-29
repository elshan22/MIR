import math

import numpy as np
from collections import Counter


class Scorer:    
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The index to score the documents with.
        number_of_documents : int
            The number of documents in the index.
        """
        self.index = index
        self.idf = {}
        self.N = number_of_documents

    def get_list_of_documents(self, query):
        """
        Returns a list of documents that contain at least one of the terms in the query.

        Parameters
        ----------
        query: List[str]
            The query to be scored

        Returns
        -------
        list
            A list of documents that contain at least one of the terms in the query.
        
        Note
        ---------
            The current approach is not optimal but we use it due to the indexing structure of the dict we're using.
            If we had pairs of (document_id, tf) sorted by document_id, we could improve this.
                We could initialize a list of pointers, each pointing to the first element of each list.
                Then, we could iterate through the lists in parallel.
            
        """
        list_of_documents = []
        for term in query:
            if term in self.index.keys():
                list_of_documents.extend(self.index[term].keys())
        return list(set(list_of_documents))
    
    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.

        Parameters
        ----------
        term : str
            The term to get the inverse document frequency for.

        Returns
        -------
        float
            The inverse document frequency of the term.
        
        Note
        -------
            It was better to store dfs in a separate dict in preprocessing.
        """
        idf = self.idf.get(term, None)
        if idf is None:
            self.idf[term] = np.log2((self.N + 0.5) / (len(self.index[term]) + 0.5)) if term in self.index else 0
            idf = self.idf[term]
        return idf
    
    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.

        Parameters
        ----------
        query : List[str]
            The query to get the term frequencies for.

        Returns
        -------
        dict
            A dictionary of the term frequencies of the terms in the query.
        """
        return dict(Counter(query))

    def compute_scores_with_vector_space_model(self, query, method):
        """
        compute scores with vector space model

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c))
            The method to use for searching.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """
        score = {}
        documents = self.get_list_of_documents(query)
        for doc_id in documents:
            score[doc_id] = self.get_vector_space_model_score(query, self.get_query_tfs(query), doc_id, method[:3], method[4:])
        return score

    def get_vector_space_model_score(self, query, query_tfs, document_id, document_method, query_method):
        """
        Returns the Vector Space Model score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        query_tfs : dict
            The term frequencies of the terms in the query.
        document_id : str
            The document to calculate the score for.
        document_method : str (n|l)(n|t)(n|c)
            The method to use for the document.
        query_method : str (n|l)(n|t)(n|c)
            The method to use for the query.

        Returns
        -------
        float
            The Vector Space Model score of the document for the query.
        """
        query_vector = []
        doc_vector = []
        for term, tf_doc in self.index.items():
            if term not in query: query_vector.append(0)
            else:
                query_tf = query_tfs[term] if query_method[0] == 'n' else 1 + np.log2(query_tfs[term])
                query_df = 1 if query_method[0] == 'n' else self.get_idf(term)
                query_vector.append(query_df * query_tf)
            if document_id not in self.index[term]: doc_vector.append(0)
            else:
                doc_tf = self.index[term][document_id] if document_method[0] == 'n' else 1 + np.log2(self.index[term][document_id])
                doc_df = 1 if document_method[0] == 'n' else self.get_idf(term)
                doc_vector.append(doc_tf * doc_df)
        if query_method[2] == 'c':
            query_vector = np.array(query_vector) / np.linalg.norm(query_vector)
        if doc_vector[2] == 'c':
            doc_vector = np.array(doc_vector) / np.linalg.norm(doc_vector)
        return np.dot(query_vector, doc_vector)

    def compute_socres_with_okapi_bm25(self, query, average_document_field_length, document_lengths):
        """
        compute scores with okapi bm25

        Parameters
        ----------
        query: List[str]
            The query to be scored
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        
        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """
        score = {}
        documents = self.get_list_of_documents(query)
        for doc_id in documents:
            score[doc_id] = self.get_okapi_bm25_score(query, doc_id, average_document_field_length, document_lengths)
        return score

    def get_okapi_bm25_score(self, query, document_id, average_document_field_length, document_lengths):
        """
        Returns the Okapi BM25 score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        document_id : str
            The document to calculate the score for.
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        float
            The Okapi BM25 score of the document for the query.
        """
        k1 = 1.5
        b = 0.75
        result = 0
        for term in query:
            if term not in self.index: continue
            result += self.get_idf(term) * (k1 + 1) * self.index[term][document_id] / (k1 * (1 + b *
                (document_lengths[document_id]/average_document_field_length - 1)) + self.index[term][document_id])
        return result

    def compute_scores_with_unigram_model(
        self, query, smoothing_method, document_lengths=None, alpha=0.5, lamda=0.5
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            A dictionary of the document IDs and their scores.
        """
        return {doc: self.compute_score_with_unigram_model(query, doc, smoothing_method, document_lengths, alpha, lamda) for doc in document_lengths.keys()}

    def compute_score_with_unigram_model(
        self, query, document_id, smoothing_method, document_lengths, alpha, lamda
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        document_id : str
            The document to calculate the score for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lamda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.

        Returns
        -------
        float
            The Unigram score of the document for the query.
        """
        score = 0
        collection = {term: sum(self.index[term].values()) for term in self.index}
        collection_size = sum(collection.values())
        for term in query.split():
            if smoothing_method == 'bayes':
                score += math.log((self.index[term][document_id] + alpha) / (document_lengths[document_id] + alpha * len(self.index)))
            elif smoothing_method == 'naive':
                score += math.log((self.index[term][document_id] + 1) / (document_lengths[document_id] + len(self.index)))
            else:
                score += (lamda * (self.index[term][document_id] / document_lengths[document_id]) +\
                                      (1 - lamda) * (collection[term] / collection_size))
        return score
