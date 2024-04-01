import numpy as np
from scipy.spatial.distance import cdist
from time import time


def load_data(data_path, label_path):
    data = np.loadtxt(data_path, delimiter=',')
    labels = np.loadtxt(label_path, delimiter=',')
    return data, labels


def cosine_similarity(X, Y):
    X_norm = np.linalg.norm(X, axis=1, keepdims=True)
    Y_norm = np.linalg.norm(Y, axis=1, keepdims=True)
    return np.dot(X, Y.T) / (X_norm * Y_norm.T)


def generalized_jaccard_similarity(X, Y):
    min_intersection = np.minimum(X[:, np.newaxis, :], Y[np.newaxis, :, :]).sum(axis=2)
    max_union = np.maximum(X[:, np.newaxis, :], Y[np.newaxis, :, :]).sum(axis=2)
    return min_intersection / max_union


def kmeans(X, n_clusters=10, distance_metric='euclidean', max_iter=10000):
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], n_clusters, replace=False), :]
    previous_centroids = centroids.copy()
    labels = np.zeros(X.shape[0])
    sse = 0
    iterations = 0
    start_time = time()

    for i in range(max_iter):
        if distance_metric == 'euclidean':
            distances = cdist(X, centroids, 'euclidean')
        elif distance_metric == 'cosine':
            distances = 1 - cosine_similarity(X, centroids)
        elif distance_metric == 'jaccard':
            distances = 1 - generalized_jaccard_similarity(X, centroids)
        else:
            raise ValueError("Unsupported distance metric")

        labels = np.argmin(distances, axis=1)
        new_centroids = np.array([X[labels == j].mean(axis=0) for j in range(n_clusters)])

        if np.all(new_centroids == previous_centroids):
            break

        new_sse = np.sum((X - centroids[labels]) ** 2)

        if i > 0 and new_sse > sse:
            print("SSE increased, stopping.")
            break

        sse = new_sse
        centroids = new_centroids
        previous_centroids = centroids.copy()
        iterations += 1

    duration = time() - start_time
    return centroids, labels, iterations, sse, duration


data_path = 'data.csv'  # Update this path
label_path = 'label.csv'  # Update this path

data, labels = load_data(data_path, label_path)

metrics = ['euclidean', 'cosine', 'jaccard']
results = {}

for metric in metrics:
    _, _, iterations, sse, duration = kmeans(data, distance_metric=metric)
    results[metric] = {
        'Iterations': iterations,
        'SSE': sse,
        'Duration': duration
    }

for metric, result in results.items():
    print(
        f"Metric: {metric}, Iterations: {result['Iterations']}, SSE: {result['SSE']}, Time: {result['Duration']} seconds")
