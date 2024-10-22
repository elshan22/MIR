import matplotlib.pyplot as plt
import numpy as np
import random
import operator
import wandb

from typing import List, Tuple
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from collections import Counter
from Logic.core.clustering.clustering_metrics import *
import plotly.figure_factory as ff


class ClusteringUtils:

    def cluster_kmeans(self, emb_vecs: List, n_clusters: int, max_iter: int) -> Tuple[List, List]:
        """
        Clusters input vectors using the K-means method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List]
            Two lists:
            1. A list containing the cluster centers.
            2. A list containing the cluster index for each input vector.
        """
        emb_vecs = np.array(emb_vecs)
        cluster_centers = emb_vecs[np.random.choice(emb_vecs.shape[0], n_clusters)]
        for _ in range(max_iter):
            cluster_indices = np.array([np.argmin(np.linalg.norm(point - cluster_centers, axis=1)) for point in emb_vecs])
            new_centers = np.array([emb_vecs[cluster_indices == i].mean(axis=0) for i in range(n_clusters)])
            if np.allclose(cluster_centers, new_centers): break
            cluster_centers = new_centers
        return cluster_centers, cluster_indices

    def get_most_frequent_words(self, documents: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Finds the most frequent words in a list of documents.

        Parameters
        -----------
        documents: List[str]
            A list of documents, where each document is a string representing a list of words.
        top_n: int, optional
            The number of most frequent words to return. Default is 10.

        Returns
        --------
        List[Tuple[str, int]]
            A list of tuples, where each tuple contains a word and its frequency, sorted in descending order of frequency.
        """
        return Counter([word for doc in documents for word in doc.split()]).most_common(top_n)

    def cluster_kmeans_WCSS(self, emb_vecs: List[List[float]], n_clusters: int, max_iter: int) -> Tuple[List[List[float]], List[int], float]:
        """
        This function performs K-means clustering on a list of input vectors and calculates the Within-Cluster Sum of Squares (WCSS) for the resulting clusters.

        Parameters
        -----------
        emb_vecs: List[List[float]]
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List[List[float]], List[int], float]
            Three elements:
            1) A list containing the cluster centers.
            2) A list containing the cluster index for each input vector.
            3) The Within-Cluster Sum of Squares (WCSS) value for the clustering.
        """
        cluster_centers, cluster_indices = self.cluster_kmeans(emb_vecs, n_clusters, max_iter)
        wcss = 0
        for i, center in enumerate(cluster_centers):
            wcss += np.sum((emb_vecs[cluster_indices == i] - center) ** 2)
        return cluster_centers, cluster_indices, wcss

    def cluster_hierarchical_single(self, emb_vecs: List, n_clusters) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with single linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        return AgglomerativeClustering(n_clusters, linkage='single').fit_predict(np.array(emb_vecs))

    def cluster_hierarchical_complete(self, emb_vecs: List, n_clusters) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with complete linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        return AgglomerativeClustering(n_clusters=n_clusters, linkage='complete').fit_predict(np.array(emb_vecs))

    def cluster_hierarchical_average(self, emb_vecs: List, n_clusters) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with average linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        return AgglomerativeClustering(n_clusters=n_clusters, linkage='average').fit_predict(np.array(emb_vecs))

    def cluster_hierarchical_ward(self, emb_vecs: List, n_clusters) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with Ward's method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        return AgglomerativeClustering(n_clusters=n_clusters, linkage='ward').fit_predict(np.array(emb_vecs))

    def visualize_kmeans_clustering_wandb(self, data, n_clusters, project_name, run_name):
        """ This function performs K-means clustering on the input data and visualizes the resulting clusters by logging a scatter plot to Weights & Biases (wandb).

        This function applies the K-means algorithm to the input data and generates a scatter plot where each data point is colored according to its assigned cluster.
        For visualization use convert_to_2d_tsne to make your scatter plot 2d and visualizable.
        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform K-means clustering on the input data with the specified number of clusters.
        3. Obtain the cluster labels for each data point from the K-means model.
        4. Create a scatter plot of the data, coloring each point according to its cluster label.
        5. Log the scatter plot as an image to the wandb run, allowing visualization of the clustering results.
        6. Close the plot display window to conserve system resources (optional).

        Parameters
        -----------
        data: np.ndarray
            The input data to perform K-means clustering on.
        n_clusters: int
            The number of clusters to form during the K-means clustering process.
        project_name: str
            The name of the wandb project to log the clustering visualization.
        run_name: str
            The name of the wandb run to log the clustering visualization.

        Returns
        --------
        None
        """
        wandb.init(project=project_name, name=run_name)
        cluster_indices = self.cluster_kmeans(data.tolist(), n_clusters, 100)[1]
        plt.scatter(data[:, 0], data[:, 1], c=cluster_indices)
        wandb.log({"K-means Clustering": plt})
        plt.close()

    def wandb_plot_hierarchical_clustering_dendrogram(self, data, project_name, linkage_method, run_name):
        """ This function performs hierarchical clustering on the provided data and generates a dendrogram plot, which is then logged to Weights & Biases (wandb).

        The dendrogram is a tree-like diagram that visualizes the hierarchical clustering process. It shows how the data points (or clusters) are progressively merged into larger clusters based on their similarity or distance.

        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform hierarchical clustering on the input data using the specified linkage method.
        3. Create a linkage matrix, which represents the merging of clusters at each step of the hierarchical clustering process.
        4. Generate a dendrogram plot using the linkage matrix.
        5. Log the dendrogram plot as an image to the wandb run.
        6. Close the plot display window to conserve system resources.

        Parameters
        -----------
        data: np.ndarray
            The input data to perform hierarchical clustering on.
        linkage_method: str
            The linkage method for hierarchical clustering. It can be one of the following: "average", "ward", "complete", or "single".
        project_name: str
            The name of the wandb project to log the dendrogram plot.
        run_name: str
            The name of the wandb run to log the dendrogram plot.

        Returns
        --------
        None
        """
        wandb.init(project=project_name, name=run_name)
        linkage_matrix = linkage(data, linkage_method)
        fig = ff.create_dendrogram(linkage_matrix, 'left')
        wandb.log({"dendrogram": fig})

    def plot_kmeans_cluster_scores(self, embeddings: List, true_labels: List, k_values: List[int], project_name=None,
                                   run_name=None):
        """ This function, using implemented metrics in clustering_metrics, calculates and plots both purity scores and silhouette scores for various numbers of clusters.
        Then using wandb plots the respective scores (each with a different color) for each k value.

        Parameters
        -----------
        embeddings : List
            A list of vectors representing the data points.
        true_labels : List
            A list of ground truth labels for each data point.
        k_values : List[int]
            A list containing the various values of 'k' (number of clusters) for which the scores will be calculated.
            Default is range(2, 9), which means it will calculate scores for k values from 2 to 8.
        project_name : str
            Your wandb project name. If None, the plot will not be logged to wandb. Default is None.
        run_name : str
            Your wandb run name. If None, the plot will not be logged to wandb. Default is None.

        Returns
        --------
        None
        """
        cm = ClusteringMetrics()
        silhouette_scores = []
        purity_scores = []
        for k in k_values:
            cluster_indices = self.cluster_kmeans(embeddings, k, 100)[1]
            silhouette_scores.append(silhouette_score(embeddings, cluster_indices))
            purity_scores.append(cm.purity_score(true_labels, cluster_indices))
        plt.plot(k_values, silhouette_scores)
        plt.plot(k_values, purity_scores)
        plt.legend()
        if project_name and run_name:
            wandb.init(project=project_name, name=run_name)
            wandb.log({"Cluster Scores": plt})
        plt.show()

    def visualize_elbow_method_wcss(self, embeddings: List, k_values: List[int], project_name: str, run_name: str):
        """ This function implements the elbow method to determine the optimal number of clusters for K-means clustering based on the Within-Cluster Sum of Squares (WCSS).

        The elbow method is a heuristic used to determine the optimal number of clusters in K-means clustering. It involves plotting the WCSS values for different values of K (number of clusters) and finding the "elbow" point in the curve, where the marginal improvement in WCSS starts to diminish. This point is considered as the optimal number of clusters.

        The function performs the following steps:
        1. Iterate over the specified range of K values.
        2. For each K value, perform K-means clustering using the `cluster_kmeans_WCSS` function and store the resulting WCSS value.
        3. Create a line plot of WCSS values against the number of clusters (K).
        4. Log the plot to Weights & Biases (wandb) for visualization and tracking.

        Parameters
        -----------
        embeddings: List
            A list of vectors representing the data points to be clustered.
        k_values: List[int]
            A list of K values (number of clusters) to explore for the elbow method.
        project_name: str
            The name of the wandb project to log the elbow method plot.
        run_name: str
            The name of the wandb run to log the elbow method plot.

        Returns
        --------
        None
        """
        wandb.init(project=project_name, name=run_name)
        wcss_values = []
        for k in k_values:
            wcss_values.append(self.cluster_kmeans_WCSS(embeddings, k, 100)[2])
        plt.plot(k_values, wcss_values)
        wandb.log({"Elbow Method": plt})
        plt.close()

    def plot_clusters(self, emb_vecs: List[List[float]], cluster_indices: List[int],
                      cluster_centers: List[List[float]]):
        emb_vecs = np.array(emb_vecs)
        cluster_centers = np.array(cluster_centers)
        for cluster in np.unique(cluster_indices):
            points = emb_vecs[np.where(cluster_indices == cluster)]
            plt.scatter(points[:, 0], points[:, 1])
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=300)
        plt.legend()
        plt.show()
