from collections import defaultdict
from queue import PriorityQueue

class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        shingles = set()
        word = '$' + word + '$'
        if len(word) < k - 1:
            return {word}
        for i in range(len(word) - k + 1):
            shingles.add(word[i: i+k])
        return shingles
    
    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """
        return len(first_set & second_set) / len(first_set | second_set)

    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = dict()
        word_counter = defaultdict(lambda: [0 for _ in range(len(all_documents))])

        for i, doc in enumerate(all_documents):
            for word in doc.split():
                if word not in all_shingled_words:
                    all_shingled_words[word] = self.shingle_word(word)
                word_counter[word][i] += 1

        return all_shingled_words, word_counter
    
    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : str
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        top5_candidates = list()
        top5 = PriorityQueue()
        top5.maxsize = 5
        word_shingles = self.shingle_word(word)
        for correct, shingles in self.all_shingled_words.items():
            if abs(len(correct) - len(word)) > 2: continue
            score = self.jaccard_score(word_shingles, shingles)
            if not top5.full():
                top5.put((score, correct))
            else:
                s, c = top5.get()
                if s < score:
                    top5.put((score, correct))
                else:
                    top5.put((s, c))
        while not top5.empty():
            top5_candidates.append(top5.get()[1])
        return top5_candidates

    def word_spell_check(self, word):
        candidates = self.find_nearest_words(word)
        tf_scores = list(map(lambda x: sum(self.word_counter[x]), candidates))
        tf_scores = list(map(lambda x: x / max(tf_scores), tf_scores))
        scores = list(map(lambda i: (
        tf_scores[i] * self.jaccard_score(self.all_shingled_words[candidates[i]], self.shingle_word(word)),
        candidates[i]), range(len(candidates))))
        return list(sorted(scores, reverse=True))[0][1]

    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : str
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        return ' '.join(list(map(self.word_spell_check, query.split())))
