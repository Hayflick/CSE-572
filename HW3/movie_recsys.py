import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from surprise import Dataset, Reader, SVD, KNNBasic
from surprise.model_selection import cross_validate
from surprise.accuracy import rmse, mae

# Load the dataset
reader = Reader(line_format='user item rating timestamp', sep=',', rating_scale=(1, 5), skip_lines=1)
data = Dataset.load_from_file('ratings_small.csv', reader=reader)

# Define algorithms
algorithms = {
    'PMF': SVD(biased=False),
    'UserCF': KNNBasic(sim_options={'user_based': True}),
    'ItemCF': KNNBasic(sim_options={'user_based': False}),
}

# Function to perform cross validation and compute average RMSE and MAE
def evaluate_algorithms(algorithms, data):
    results = {}
    for name, algo in algorithms.items():
        cv_result = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
        avg_rmse = np.mean(cv_result['test_rmse'])
        avg_mae = np.mean(cv_result['test_mae'])
        results[name] = (avg_rmse, avg_mae)
    return results

initial_results = evaluate_algorithms(algorithms, data)
print(initial_results)

# Investigate different similarity metrics
similarity_options = ['cosine', 'msd', 'pearson']
user_cf_metrics_results = defaultdict(list)
item_cf_metrics_results = defaultdict(list)

for metric in similarity_options:
    user_cf = KNNBasic(sim_options={'name': metric, 'user_based': True})
    item_cf = KNNBasic(sim_options={'name': metric, 'user_based': False})
    user_results = cross_validate(user_cf, data, measures=['RMSE'], cv=5, verbose=True)
    item_results = cross_validate(item_cf, data, measures=['RMSE'], cv=5, verbose=True)
    user_cf_metrics_results['RMSE'].append(np.mean(user_results['test_rmse']))
    item_cf_metrics_results['RMSE'].append(np.mean(item_results['test_rmse']))

# Plotting the impact of similarity metrics
plt.figure(figsize=(10, 6))
plt.plot(similarity_options, user_cf_metrics_results['RMSE'], marker='o', label='User-based CF')
plt.plot(similarity_options, item_cf_metrics_results['RMSE'], marker='o', label='Item-based CF')
plt.title('Impact of Similarity Metrics on RMSE')
plt.xlabel('Similarity Metrics')
plt.ylabel('Average RMSE')
plt.legend()
plt.show()

# Look at the impact of the number of neighbors (easy peasy)
ks = range(1, 101, 5)
user_cf_k_results = defaultdict(list)
item_cf_k_results = defaultdict(list)

for k in ks:
    user_cf = KNNBasic(k=k, sim_options={'user_based': True})
    item_cf = KNNBasic(k=k, sim_options={'user_based': False})
    user_results = cross_validate(user_cf, data, measures=['RMSE'], cv=5, verbose=True)
    item_results = cross_validate(item_cf, data, measures=['RMSE'], cv=5, verbose=True)
    user_cf_k_results['RMSE'].append(np.mean(user_results['test_rmse']))
    item_cf_k_results['RMSE'].append(np.mean(item_results['test_rmse']))

# Plotting the impact of the number of neighbors
plt.figure(figsize=(10, 6))
plt.plot(ks, user_cf_k_results['RMSE'], marker='o', label='User-based CF')
plt.plot(ks, item_cf_k_results['RMSE'], marker='o', label='Item-based CF')
plt.title('Impact of Number of Neighbors on RMSE')
plt.xlabel('Number of Neighbors (k)')
plt.ylabel('Average RMSE')
plt.xscale('log')
plt.legend()
plt.show()
