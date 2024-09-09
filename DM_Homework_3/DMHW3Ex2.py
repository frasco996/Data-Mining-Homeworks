import numpy as np
from random import gauss
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import time

def generate_dataset(k, n, d, sigma):
    try:
        data = np.ndarray(shape=(k*n, d+k), dtype=float, order='F')

        for i in range(k*n):
            for j in range(d):
                data[i][k+j] = gauss(0, sigma)

            for j in range(k):
                if i % k == j:
                    data[i][j] = 1
                else:
                    data[i][j] = 0
        return data

    except MemoryError as e:
        print(f"MemoryError: {e}")
        # Handle the error, for example, append parameters and exception to a file
        append_error_to_file(k, n, d, sigma, e)
        return None # Reraise the exception to stop the program or handle it at a higher level

def append_error_to_file(k, n, d, sigma, exception):
    output_file_path = 'Performance_report.txt'
    with open(output_file_path, 'a') as file:
        file.write(f"MemoryError occurred:\n")
        file.write(f"Parameters: k = {k}, n = {n}, d = {d}, sigma = {sigma}\n")
        file.write(f"Exception: {exception}\n")
        file.write("-" * 80 + "\n")


def ground_clustering(data,labels,n,k):
    c=0
    for i in range(k):
        for j in range(n):
            if labels[i] == labels[i+(j*k)]:
                c=c+1
    res=(c/(n*k))*100
    return res

# Set parameters
k = 50
n = 1000
d = k
sigma = 1/k
m = 50
# k_values = [50, 100, 200]
# n_values = [1000, 10000, 100000]
# d_values = [k, 100*k, 100*k^2]  # d = k, 100k, 100k^2
# sigma_values = [1/k, 1/np.sqrt(k), 0.5]

# Timing for dataset generation
start_time_generate = time.time()
dataset = generate_dataset(k, n, d, sigma)
if dataset is not None:
    end_time_generate = time.time()
    generate_time = end_time_generate - start_time_generate
    print(f"Parameters: k = {k}, n = {n}, d = {d}, sigma = {sigma}")
    print(f"Dataset generation time: {generate_time:.4f} seconds")

    # Timing for KMeans fitting
    start_time_kmeans = time.time()
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init="auto",max_iter=300)
    kmeans.fit(dataset)
    end_time_kmeans = time.time()
    kmeans_time = end_time_kmeans - start_time_kmeans

    labelsK=kmeans.labels_
    resK=ground_clustering(dataset,labelsK,n,k)
    print(f"KMeans fitting time without PCA: {kmeans_time:.4f} seconds")
    print("Accuracy K: ",resK)
    start_time_pca = time.time()
    pca = PCA(n_components=m)
    projected_data = pca.fit_transform(dataset)
    end_time_pca = time.time()
    pca_time = end_time_pca - start_time_pca

    print(f"PCA time: {pca_time:.4f} seconds")

    # Timing for KMeans fitting
    start_time_kmeans = time.time()
    kmeansPCA = KMeans(n_clusters=k, init='k-means++', n_init="auto",max_iter=300)
    kmeans.fit(projected_data)
    end_time_kmeans = time.time()
    kmeans_pca_time = end_time_kmeans - start_time_kmeans

    labelsPK=kmeans.labels_
    resPK=ground_clustering(dataset,labelsPK,n,k)
    print("Accuracy PK: ",resPK)
    print(f"KMeans fitting time with PCA: {kmeans_pca_time:.4f} seconds")


    output_file_path = 'Performance_report.txt'
    with open(output_file_path, 'a') as file:
        file.write(f"Parameters: k = {k}, n = {n}, d = {d}, sigma = {sigma}\n")
        file.write(f"Dataset generation time: {generate_time:.4f} seconds\n")
        file.write(f"KMeans fitting time without PCA: {kmeans_time:.4f} seconds\n")
        file.write(f"Accuracy : {resK}\n")
        file.write(f"PCA time: {pca_time:.4f} seconds\n")
        file.write(f"KMeans fitting time with PCA: {kmeans_pca_time:.4f} seconds\n")
        file.write(f"Accuracy : {resPK}\n")
        file.write("-" * 80 + "\n") 
    