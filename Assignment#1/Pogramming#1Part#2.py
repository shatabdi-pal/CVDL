#Programming Assignment#1 Part#2
#Implemenation of K-Means clustering Algorithm


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image


#reading .txt file into dataframe
data = np.loadtxt("510_cluster_dataset.txt")

#visulization of initial dataset
df = pd.DataFrame({'x': data[:,0], 'y': data[:,1]})
plt.scatter(df['x'], df['y'])
plt.show()

def k_means(X, K, max_iters=100, r=10):
    best_sse = np.inf
    best_centriods = None

    for i in range(r):
        # Initialize cluster centriods randomly
        centriods = X[np.random.choice(range(len(X)), size=K, replace=False)]

        for j in range(max_iters):
            # Assign each data point to the nearest cluster
            distances = np.linalg.norm(X[:, np.newaxis] - centriods, axis=2)
            clusters = np.argmin(distances, axis=1)

            # Update cluster centriods
            new_centriods = np.array([X[clusters == k].mean(axis=0) for k in range(K)])

            # Check convergence
            if np.all(centriods == new_centriods):
                break

            centriods = new_centriods

        # Calculate the sum of squares error (SSE)
        sse = np.sum(np.square(X - centriods[clusters]), axis=1).sum()

        # Keep track of the best SSE and corresponding cluster centriods
        if sse < best_sse:
            best_sse = sse
            best_centriods = centriods

    return best_centriods, clusters, best_sse





# Run K-Means for K=2, K=3, and K=4
k_values = [2, 3, 4]
errors = []

plt.figure(figsize=(12, 4))

for i, K in enumerate(k_values):
    centriods, clusters, sse = k_means(data, K)
    errors.append(sse)

    plt.subplot(1, 3, i + 1)
    plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis')
    plt.scatter(centriods[:, 0], centriods[:, 1], c='red', marker='X', s=100)
    plt.title(f'K = {K}, SSE = {sse:.2f}')

plt.tight_layout()
plt.show()

# Print the sum of squares error for each K value
for i, K in enumerate(k_values):
    print(f'K = {K}, SSE = {errors[i]:.2f}')


def image_k_means(image, K, max_iters=100, r=10):
    # Convert the image to RGB array
    image_rgb = np.array(image.convert("RGB"))

    # Reshape the RGB array to 2D
    image_2d = image_rgb.reshape(-1, 3)

    # Run K-Means on the image
    centriods, clusters, sse = k_means(image_2d, K, max_iters, r)

    # Reshape the clusters back to 2D
    image_clusters = clusters.reshape(image_rgb.shape[:2])

    return centriods, image_clusters, sse


# Load the images
image1 = Image.open("Kmean_img1.jpg")
image2 = Image.open("Kmean_img2.jpg")

# Run K-Means for K=5 and K=10 on the images
k_values_images = [5, 10]
errors_images = []

for i, K in enumerate(k_values_images):
    centriods, image_clusters, sse = image_k_means(image1, K)
    errors_images.append(sse)

    plt.figure()
    plt.imshow(image_clusters, cmap='viridis')
    plt.title(f'K = {K}, SSE = {sse:.2f}')
    plt.axis('off')
    plt.show()

# Print the sum of squares error for each K value
for i, K in enumerate(k_values_images):
    print(f'K = {K}, SSE = {errors_images[i]:.2f}')
