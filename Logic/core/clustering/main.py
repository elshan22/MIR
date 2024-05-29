import time

import numpy as np
from tqdm import tqdm

from Logic.core.word_embedding.fasttext_data_loader import FastTextDataLoader
from Logic.core.word_embedding.fasttext_model import FastText
from Logic.core.clustering.dimension_reduction import DimensionReduction
from Logic.core.clustering.clustering_metrics import ClusteringMetrics
from Logic.core.clustering.clustering_utils import ClusteringUtils
from collections import Counter

import sys
sys.setrecursionlimit(20000)
tqdm.pandas()


def plot_hierarchy(X, method):
    X_list = X.tolist()
    y_single = clustering_utils.cluster_hierarchical_single(X_list, n_cluster)
    y_complete = clustering_utils.cluster_hierarchical_complete(X_list, n_cluster)
    y_average = clustering_utils.cluster_hierarchical_average(X_list, n_cluster)
    y_ward = clustering_utils.cluster_hierarchical_ward(X_list, n_cluster)
    clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(X, 'Phase2', 'single', f'hierarchical_single_{method}')
    clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(X, 'Phase2', 'complete', f'hierarchical_complete_{method}')
    clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(X, 'Phase2', 'average', f'hierarchical_average_{method}')
    clustering_utils.wandb_plot_hierarchical_clustering_dendrogram(X, 'Phase2', 'ward', f'hierarchical_ward_{method}')
    return y_single, y_complete, y_average, y_ward


def print_accuracy(y, y_pred, X, method):
    clustering_metrics = ClusteringMetrics()
    print(f'{method}:')
    print(f'Adjusted rand score: {clustering_metrics.adjusted_rand_score(y, y_pred)}')
    print(f'Purity score: {clustering_metrics.purity_score(y, y_pred)}')
    print(f'Silhouette score: {clustering_metrics.silhouette_score(X, y_pred)}')


if __name__ == '__main__':
    # 0. Embedding Extraction
    fasttext_dl = FastTextDataLoader('../IMDB_crawled.json')
    X, y = fasttext_dl.create_train_data()
    fasttext = FastText()
    fasttext.prepare(None, 'load', path='../FastText_model.bin')
    X = [fasttext.model.get_sentence_vector(i) for i in X]
    X_list = X.copy()
    X = np.array(X)
    # 1. Dimension Reduction Perform Principal Component Analysis (PCA): - Reduce the dimensionality of features using
    # PCA. (you can use the reduced feature afterward or use to the whole embeddings) - Find the Singular Values and use
    # the explained_variance_ratio_ attribute to determine the percentage of variance explained by each principal
    # component. - Draw plots to visualize the results.
    dimension_reduction = DimensionReduction()
    X_reduced = dimension_reduction.pca_reduce_dimension(X, n_components=5)
    dimension_reduction.wandb_plot_explained_variance_by_components(X, 'Phase2', 'explained variance by components')
    # Implement t-SNE (t-Distributed Stochastic Neighbor Embedding): - Create the convert_to_2d_tsne function,
    # which takes a list of embedding vectors as input and reduces the dimensionality to two dimensions using the t-SNE
    # method. - Use the output vectors from this step to draw the diagram.
    X_2d = dimension_reduction.convert_to_2d_tsne(X)
    dimension_reduction.wandb_plot_2d_tsne(X, 'Phase2', 't-SNE')
    # 2. Clustering
    ## K-Means Clustering
    # Implement the K-means clustering algorithm from scratch.
    #  Create document clusters using K-Means.
    # Run the algorithm with several different values of k.
    # For each run:
    #     - Determine the genre of each cluster based on the number of documents in each cluster.
    #     - Draw the resulting clustering using the two-dimensional vectors from the previous section.
    #     - Check the implementation and efficiency of the algorithm in clustering similar documents.
    # Draw the silhouette score graph for different values of k and perform silhouette analysis to choose the appropriate k.
    # Plot the purity value for k using the labeled data and report the purity value for the final k. (Use the provided functions in utilities)
    n_cluster = len(np.unique(y))
    clustering_utils = ClusteringUtils()
    range_k = [5, 10, 15, 20, 25]
    clustering_metrics = ClusteringMetrics()
    for k in range_k:
        start_time = time.time()
        cluster_centers, cluster_indices, wcss = clustering_utils.cluster_kmeans_WCSS(X_list, k, 100)
        end_time = time.time()
        print(f"Runtime of K-means with k = {k}: {end_time - start_time}")
        print(f"Predicted genres for k = {k}: {Counter(cluster_indices)}")
        clustering_utils.plot_clusters(X_list, cluster_indices, cluster_centers)
        print(f"Adjusted rand score for k= {k}: {clustering_metrics.adjusted_rand_score(y, cluster_indices)}")
        print(f"Purity score for k = {k}: {clustering_metrics.purity_score(y, cluster_indices)}")
        print(f"Silhouette score score for k = {k}: {clustering_metrics.silhouette_score(X_list, cluster_indices)}")
    clustering_utils.visualize_elbow_method_wcss(X_list, range_k, 'Phase2', 'elbow_method_wcss')
    ## Hierarchical Clustering
    # Perform hierarchical clustering with all different linkage methods.
    # Visualize the results.
    X_sample = X[np.random.choice([i for i in range(len(X))], 100)]
    y_single, y_complete, y_average, y_ward = plot_hierarchy(X, 'full')
    plot_hierarchy(X_sample, 'sample')
    # 3. Evaluation
    # Using clustering metrics, evaluate how well your clustering method is performing.
    print_accuracy(y, y_single, X_list, 'SINGLE')
    print_accuracy(y, y_complete, X_list, 'COMPLETE')
    print_accuracy(y, y_average, X_list, 'AVERAGE')
    print_accuracy(y, y_ward, X_list, 'WARD')
