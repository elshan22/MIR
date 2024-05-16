import numpy as np
import math
import itertools
import random
from collections import defaultdict


class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.hash_functions = [lambda x: hash(f"function{i}:{x}") for i in range(num_hashes)]

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        words = document.split()
        shingles = set()
        for i in range(len(words) - k + 1):
            shingles.add(' '.join(words[i:i+k]))
        return shingles

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        doc_shingles = [self.shingle_document(x) for x in self.documents]
        all_shingles = set()
        for shingles in doc_shingles:
            all_shingles |= shingles
        matrix = []
        for shingle in all_shingles:
            matrix.append(np.array([1 if shingle in doc else 0 for doc in doc_shingles]))
        return np.array(matrix)

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        signature = np.zeros((len(self.documents), len(self.hash_functions)))
        for i, doc in enumerate(self.documents):
            for j, hash_func in enumerate(self.hash_functions):
                min_hash = float('inf')
                for shingle in self.shingle_document(doc):
                    hash_value = hash_func(shingle)
                    if hash_value < min_hash:
                        min_hash = hash_value
                signature[i, j] = min_hash
        return signature

    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        buckets = defaultdict(list)
        for i, sig in enumerate(signature):
            for band in range(bands):
                buckets[hash(tuple(sig[band*rows_per_band:(band+1)*rows_per_band]))].append(i)
        return buckets

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        return self.lsh_buckets(self.min_hash_signature(), len(self.hash_functions)//10)

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        return len(first_set & second_set) / len(first_set | second_set)

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0
        all_candidate_pairs = set()

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            all_candidate_pairs.update(itertools.combinations(unique_doc_ids, 2))

        for first_doc_id, second_doc_id in all_candidate_pairs:
            all_near_duplicates += 1

            first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
            second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

            near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
            current_score = 0

            for _ in range(5):
                random_doc_id = first_doc_id
                while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                    random_doc_id = random.randint(0, len(all_documents) - 1)
                random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                if near_duplicated_jaccard_score > random_jaccard_score:
                    current_score += 1

            if current_score == 5:
                correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)

if __name__ == "__main__":
    import json
    with open("LSHFakeData.json", "r") as f:
        documents = json.load(f)
    documents = [" # ".join(doc["summaries"]) for doc in documents]
    minhash_lsh = MinHashLSH(documents, 100)
    minhash_lsh.jaccard_similarity_test(minhash_lsh.perform_lsh(), documents)
