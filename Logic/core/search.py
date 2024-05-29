from collections import defaultdict

from utility.preprocess import Preprocessor
from utility.scorer import Scorer
from indexer.indexes_enum import Indexes, Index_types
from indexer.index_reader import Index_reader


class SearchEngine:
    def __init__(self):
        """
        Initializes the search engine.

        """
        path = '/indexes'
        self.document_indexes = {
            Indexes.STARS: Index_reader(path, Indexes.STARS),
            Indexes.GENRES: Index_reader(path, Indexes.GENRES),
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES)
        }
        self.tiered_index = {
            Indexes.STARS: Index_reader(path, Indexes.STARS, Index_types.TIERED),
            Indexes.GENRES: Index_reader(path, Indexes.GENRES, Index_types.TIERED),
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES, Index_types.TIERED)
        }
        self.document_lengths_index = {
            Indexes.STARS: Index_reader(path, Indexes.STARS, Index_types.DOCUMENT_LENGTH),
            Indexes.GENRES: Index_reader(path, Indexes.GENRES, Index_types.DOCUMENT_LENGTH),
            Indexes.SUMMARIES: Index_reader(path, Indexes.SUMMARIES, Index_types.DOCUMENT_LENGTH)
        }
        self.metadata_index = Index_reader(path, Indexes.DOCUMENTS, Index_types.METADATA)

    def search(
            self,
            query,
            method,
            weights,
            safe_ranking=True,
            max_results=10,
            smoothing_method=None,
            alpha=0.5,
            lamda=0.5,
    ):
        """
        searches for the query in the indexes.

        Parameters
        ----------
        query : str
            The query to search for.
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        safe_ranking : bool
            If True, the search engine will search in whole index and then rank the results. 
            If False, the search engine will search in tiered index.
        max_results : int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            A list of tuples containing the document IDs and their scores sorted by their scores.
        """

        preprocessor = Preprocessor([query])
        query = preprocessor.preprocess()[0].split()

        scores = {}
        if method == 'unigram':
            self.find_scores_with_unigram_model(query, smoothing_method, weights, scores, alpha, lamda)
        elif safe_ranking:
            self.find_scores_with_safe_ranking(query, method, weights, scores)
        else:
            self.find_scores_with_unsafe_ranking(query, method, weights, max_results, scores)

        final_scores = {}

        self.aggregate_scores(weights, scores, final_scores)

        result = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        if max_results is not None:
            result = result[:max_results]

        return result

    def aggregate_scores(self, weights, scores, final_scores):
        """
        Aggregates the scores of the fields.

        Parameters
        ----------
        weights : dict
            The weights of the fields.
        scores : dict
            The scores of the fields.
        final_scores : dict
            The final scores of the documents.
        """
        for field, field_score in scores.items():
            for doc_id, score in field_score.items():
                if doc_id not in final_scores: final_scores[doc_id] = 0
                final_scores[doc_id] += score * weights[field]

    def find_scores_with_unsafe_ranking(self, query, method, weights, max_results, scores):
        """
        Finds the scores of the documents using the unsafe ranking method using the tiered index.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        max_results : int
            The maximum number of results to return.
        scores : dict
            The scores of the documents.
        """
        for field in weights.keys():
            scores[field] = {}
            for tier in ['first_tier', 'second_tier', 'third_tier']:
                indices = self.tiered_index[field].index[tier]
                docs = set()
                for doc_tf in indices.values():
                    docs |= doc_tf.keys()
                scorer = Scorer(self.tiered_index[field].index[tier], len(docs))
                if method == 'OkapiBM25':
                    avdl = sum([x[1] for x in filter(lambda x: x[0] in docs, self.document_lengths_index[field].index.items())]) / scorer.N
                    scores[field] = self.merge_scores(scores[field],
                        scorer.compute_socres_with_okapi_bm25(query, avdl, self.document_lengths_index[field].index))
                else:
                    scores[field] = self.merge_scores(scores[field], scorer.compute_scores_with_vector_space_model(query, method))
                if scores[field] >= max_results: break

    def find_scores_with_safe_ranking(self, query, method, weights, scores):
        """
        Finds the scores of the documents using the safe ranking method.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c)) | OkapiBM25
            The method to use for searching.
        weights: dict
            The weights of the fields.
        scores : dict
            The scores of the documents.
        """
        for field in weights.keys():
            scorer = Scorer(self.document_indexes[field].index, len(self.document_lengths_index[field].index))
            if method == 'OkapiBM25':
                avdl = sum(self.document_lengths_index[field].index.values()) / scorer.N
                scores[field] = scorer.compute_socres_with_okapi_bm25(query, avdl, self.document_lengths_index[field].index)
            else:
                scores[field] = scorer.compute_scores_with_vector_space_model(query, method)

    def find_scores_with_unigram_model(
            self, query, smoothing_method, weights, scores, alpha=0.5, lamda=0.5
    ):
        """
        Calculates the scores for each document based on the unigram model.

        Parameters
        ----------
        query : str
            The query to search for.
        smoothing_method : str (bayes | naive | mixture)
            The method used for smoothing the probabilities in the unigram model.
        weights : dict
            A dictionary mapping each field (e.g., 'stars', 'genres', 'summaries') to its weight in the final score. Fields with a weight of 0 are ignored.
        scores : dict
            The scores of the documents.
        alpha : float, optional
            The parameter used in bayesian smoothing method. Defaults to 0.5.
        lambda : float, optional
            The parameter used in some smoothing methods to balance between the document
            probability and the collection probability. Defaults to 0.5.
        """
        for field in weights.keys():
            scorer = Scorer(self.document_indexes[field].index, len(self.document_lengths_index[field].index))
            scores[field] = scorer.compute_scores_with_unigram_model(query, smoothing_method, self.document_lengths_index[field].index, alpha, lamda)

    def merge_scores(self, scores1, scores2):
        """
        Merges two dictionaries of scores.

        Parameters
        ----------
        scores1 : dict
            The first dictionary of scores.
        scores2 : dict
            The second dictionary of scores.

        Returns
        -------
        dict
            The merged dictionary of scores.
        """
        result = defaultdict(dict)
        for field, field_score in list(scores1.items()) + list(scores2.items()):
            for doc_id, value in field_score.items():
                result[field][doc_id] += value
        return result


if __name__ == '__main__':
    search_engine = SearchEngine()
    query = "spider man in wonderland"
    method = "lnc.ltc"
    weights = {
        Indexes.STARS: 1,
        Indexes.GENRES: 1,
        Indexes.SUMMARIES: 1
    }
    result = search_engine.search(query, method, weights)

    print(result)
