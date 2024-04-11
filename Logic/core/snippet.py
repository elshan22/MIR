from Logic.core.preprocess import Preprocessor


class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """
        return ' '.join(['' if word in Preprocessor([]).stopwords else word for word in Preprocessor([]).tokenize(query)])

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        query = self.remove_stop_words_from_query(query)
        final_snippet = ""
        query_words = query.split()
        doc_words = doc.split()
        not_exist_words = list(filter(lambda x: x in doc_words, query_words))
        arr = [query_words.index(x) if x in query_words else -1 for x in doc_words]
        index_arr = []
        for i, n in enumerate(arr):
            if n != -1: index_arr.append(i)
        while index_arr:
            keywords, end, start = {}, -1, float('inf')
            for n in index_arr:
                s, e = max(0, n-self.number_of_words_on_each_side), min(len(arr), n+1+self.number_of_words_on_each_side)
                sub = set(arr[s:e])
                sub.discard(-1)
                if len(sub) > len(keywords) or (len(sub) == len(keywords) and e - s > end - start):
                    keywords, end, start = sub, e, s
            index_arr = list(filter(lambda x: not (arr[x] in keywords or start <= x < end), index_arr))
            arr = list(map(lambda x: -1 if x in keywords else x, arr))
            snippet = doc_words[start:end]
            for i, word in enumerate(snippet):
                if word in query_words: snippet[i] = f'***{snippet[i]}***'
            final_snippet += '...' + ' '.join(snippet)
        final_snippet = final_snippet[3:] if final_snippet[:3] == '...' else final_snippet
        return final_snippet, not_exist_words
