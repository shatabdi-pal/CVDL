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



#visulization of data after clustering
def visualize_clusters(data,cluster, centroids):
    df = pd.DataFrame(dict(x=data[:,0], y=data[:,1], label=cluster))
    # colors = {0:'blue', 1:'orange'}
    #colors = {0:'blue', 1:'orange', 2:'green'}
    colors = {0:'blue', 1:'orange', 2:'green',3:'red'}
    #colors = {0:'blue', 1:'orange', 2:'green',3:'red', 4:'purple', 5:'brown',6:'yellow',7:'pink',8:'indigo'}
    fig, ax = plt.subplots(figsize=(6, 6))
    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
    ax.scatter(centroids[:, 0], centroids[:, 1], color='red', marker='*')
    plt.show()


#deciding the stopping condition
def measure_diff(centroids_prev, centroids_new):
    result = 0
    for m,n in zip(centroids_prev,centroids_new):
        result+= distance_calculation(m,n)
    return result



def k_means_clustering(k, data):
    cluster = [0]*len(data)
    centroid_prev = initialize_centroids(k, data)
    #stopping criterion
    centroid_diff = 100
    while centroid_diff >.001:
        cluster = cluster_assignment(k, data, centroid_prev)
        visualize_clusters(data, cluster, centroid_prev)
        centroid_new = calculate_centroids(k, data, cluster)
        centroid_diff = measure_diff(centroid_new, centroid_prev)
        centroid_prev = centroid_new
    return cluster


# Suppress the UserWarning
warnings.filterwarnings("ignore", category=UserWarning)

#varying number of clusters
if __name__ == "__main__":
    # k = 2
    # cluster = k_means_clustering(k, data)
    # # k = 3
    # # cluster = k_means_clustering(k, data)
    k = 4
    cluster = k_means_clustering(k, data)
