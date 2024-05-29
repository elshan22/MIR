import json
import random

from graph import LinkGraph
from Logic.core.indexer.indexes_enum import Indexes
from Logic.core.indexer.index_reader import Index_reader


class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.document_index = Index_reader('../Indexes/', Indexes.DOCUMENTS).get_index()
        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = {}
        self.authorities = {}
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        for movie in self.root_set:
            movie_node = ('movie', movie['id'])
            self.graph.add_node(movie_node)
            self.hubs[movie_node] = 1
            if movie['stars']:
                for star in movie['stars']:
                    star_node = ('star', star)
                    self.graph.add_node(star_node)
                    self.graph.add_edge(movie_node, star_node)
                    self.graph.add_edge(star_node, movie_node)
                    self.authorities[star_node] = 1

    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        for movie in corpus:
            movie_node = ('movie', movie['id'])
            if movie_node not in self.graph.graph:
                self.graph.add_node(movie_node)
                self.hubs[movie_node] = 1
            if movie['stars']:
                for star in movie['stars']:
                    star_node = ('star', star)
                    if star_node not in self.graph.graph:
                        self.graph.add_node(star_node)
                        self.authorities[star_node] = 1
                    self.graph.add_edge(movie_node, star_node)
                    self.graph.add_edge(star_node, movie_node)

    def hits(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """
        for _ in range(num_iteration):
            new_movie_hubs = {}
            new_star_authorities = {}
            for star_node in self.authorities:
                new_star_authorities[star_node] = sum(self.hubs[movie_node] for movie_node in self.graph.get_predecessors(star_node))
            norm = sum(new_star_authorities.values())
            for star_node in new_star_authorities:
                new_star_authorities[star_node] /= norm
            for movie_node in self.hubs:
                new_movie_hubs[movie_node] = sum(self.authorities[star_node] for star_node in self.graph.get_successors(movie_node))
            norm = sum(new_movie_hubs.values())
            for movie_node in new_movie_hubs:
                new_movie_hubs[movie_node] /= norm
            self.hubs = new_movie_hubs
            self.authorities = new_star_authorities
        top_movies = sorted(self.hubs.items(), key=lambda item: item[1], reverse=True)[:max_result]
        top_stars = sorted(self.authorities.items(), key=lambda item: item[1], reverse=True)[:max_result]
        movie_titles = [self.document_index[movie[0][1]]['title'] for movie in top_movies]
        star_names = [star[0][1] for star in top_stars]
        return star_names, movie_titles


if __name__ == "__main__":
    corpus = json.load(open('../IMDB_crawled.json'))
    root_set = random.sample(corpus, len(corpus)//20)

    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=10)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')
