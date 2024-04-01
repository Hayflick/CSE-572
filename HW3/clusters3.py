import numpy as np
import pandas as pd
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from collections import Counter

class CustomKMeans:
    def __init__(self, n_clusters, max_iter=500, tol=1e-4, dist_metric='euclidean'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.dist_metric = dist_metric
        self.centroids = None
        self.labels = None
        self.sse = None

    def initialize_centroids(self, data):
        np.random.seed(42)
        self.centroids = data[np.random.choice(data.shape[0], self.n_clusters, replace=False)]

    def compute_distances(self, data):
        if self.dist_metric in ['euclidean', 'cosine']:
            return pairwise_distances(data, self.centroids, metric=self.dist_metric)
        elif self.dist_metric == 'jaccard':
            return 1 - self.generalized_jaccard_similarity(data, self.centroids)
        else:
            raise ValueError("Unsupported distance metric.")

    def generalized_jaccard_similarity(self, data, centroids):
        min_values = np.minimum.outer(data, centroids)
        max_values = np.maximum.outer(data, centroids)
        similarity = np.sum(min_values, axis=2) / np.sum(max_values, axis=2)
        return similarity

    def assign_clusters(self, distances):
        return np.argmin(distances, axis=1)

    def update_centroids(self, data, labels):
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(self.n_clusters)])
        return new_centroids

    def compute_sse(self, data, labels):
        distances = np.min(self.compute_distances(data), axis=1)
        self.sse = np.sum(distances ** 2)

    def fit(self, data):
        self.initialize_centroids(data)
        for i in range(self.max_iter):
            distances = self.compute_distances(data)
            new_labels = self.assign_clusters(distances)
            new_centroids = self.update_centroids(data, new_labels)
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                break
            self.centroids = new_centroids
            self.labels = new_labels
            self.compute_sse(data, new_labels)

def calculate_accuracy(true_labels, predicted_labels):
    label_mapping = {}
    for k in np.unique(predicted_labels):
        cluster_labels = true_labels[predicted_labels == k]
        most_common_label = Counter(cluster_labels).most_common(1)[0][0]
        label_mapping[k] = most_common_label

    mapped_predictions = np.array([label_mapping[cluster] for cluster in predicted_labels])
    accuracy = np.mean(mapped_predictions == true_labels)
    return accuracy

def load_data():
    data = pd.read_csv('data.csv', header=None).values
    labels = pd.read_csv('label.csv', header=None).squeeze().values
    return data, labels

# Load the data
data, labels = load_data()  #Don't forget to implement this
n_clusters = len(np.unique(labels))

# Fit models
results = {}
for dist_metric in ['euclidean', 'cosine', 'jaccard']:
    model = CustomKMeans(n_clusters=n_clusters, dist_metric=dist_metric)
    model.fit(data)
    accuracy = calculate_accuracy(labels, model.labels)
    results[dist_metric] = (model.sse, accuracy)

# Display results
for dist_metric, (sse, accuracy) in results.items():
    print(f"{dist_metric.capitalize()} KMeans - SSE: {sse}, Accuracy: {accuracy}")
