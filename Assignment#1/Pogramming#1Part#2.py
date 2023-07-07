#Programming Assignment#1 Part#2
#Implemenation of K-Means clustering Algorithm


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import warnings


#reading .txt file into dataframe
data = np.loadtxt("510_cluster_dataset.txt")


#visulization of initial dataset
df = pd.DataFrame({'x': data[:,0], 'y': data[:,1]})
plt.scatter(df['x'], df['y'])
plt.show()


#initialization of centriods by random values
def initialize_centroids(k, data):
    centroids = []
    for i in range(k):
        x = np.random.uniform(min(data[:,0]), max(data[:,0]))
        y = np.random.uniform(min(data[:,1]), max(data[:,1]))
        centroids.append([x, y])
    return np.asarray(centroids)


#euclidean distance calculation
def distance_calculation(xi, yi):
    distance = np.sqrt(sum(np.square(xi-yi)))
    return distance



#cluster assignment based on the distance of data point from centroids
def cluster_assignment(k, data, centroids):
    cluster = [-1]*len(data)
    for i in range(len(data)):
        dist_arr = []
        for j in range(k):
            temp = distance_calculation(data[i], centroids[j])
            dist_arr.append(temp)
        index = np.argmin(dist_arr)
        cluster[i] = index
    return np.asarray(cluster)



#updating the centroids
def calculate_centroids(k, data, cluster):
    centroid = []
    for i in range(k):
        temp = []
        for j in range(len(data)):
            if cluster[j]==i:
                temp.append(data[j])
                new_mean = np.mean(temp, axis=0)
        centroid.append(new_mean)
    return np.asarray(centroid)



#deciding the stopping condition
def measure_diff(centroids_prev, centroids_new):
    result = 0
    for m,n in zip(centroids_prev,centroids_new):
        result+= distance_calculation(m,n)
    return result



def k_means(data, k,  r):
    cluster = [0]*len(data)
    centroid_prev = initialize_centroids(k, data)
    best_error = float('inf')
    best_centroids = None
    prev_error = None

    for _ in range(r):
        while True:
            cluster = cluster_assignment(k, data, centroid_prev)
            centroid_new = calculate_centroids(k, data, cluster)
            centroid_diff = measure_diff(centroid_new, centroid_prev)
            # Calculate the sum of squares error
            error = centroid_diff
            if prev_error is not None and error >= prev_error:
                break

            prev_error = error
            centroid_prev = centroid_new

        if error < best_error:
            best_error = error
            best_centroids = centroid_new

    return best_centroids, best_error




k_values = [2, 3, 4]
results = []

for k in k_values:
    centroids, error = k_means(data, k, r=10)
    results.append((centroids, error))

# Plot the results
plt.figure(figsize=(12, 6))

for i, (centroids, _) in enumerate(results):
    plt.subplot(1, len(k_values), i+1)
    plt.scatter(data[:, 0], data[:, 1], c=np.random.rand(len(data)), alpha=0.5)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
    plt.title(f'K = {len(centroids)}')
    plt.xlabel('X')
    plt.ylabel('Y')

plt.tight_layout()
plt.show()

# Print the sum of squares error for each model
for i, (_, error) in enumerate(results):
    print(f'Sum of Squares Error (K={k_values[i]}): {error}')

