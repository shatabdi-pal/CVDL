#Implementation for convolution,maxpooling and sigmoid activatio
import numpy as np

# Given input matrix I
I = np.array([[2, 0, 3, 4, 1, 7],
              [15, 12, 2, 1, 0, 5],
              [14, 11, 1, 0, 1, 3],
              [12, 10, 0, 0, 0, 1],
              [11, 8, 1, 0, 0, 4],
              [7, 5, 1, 2, 3, 6]])

# Given filters and biases
f1_1 = np.array([[1, 1, 1],
                 [0, 0, 0],
                 [-1, -1, -1]])
b1_1 = 3

f1_2 = np.array([[1, 0, 0],
                 [0, 1, 0],
                 [0, 0, 1]])
b1_2 = -1

# Function for 2D convolution operation
def convolve2D(image, filter):
    return np.rot90(np.array([[np.sum(np.multiply(image[i:i+filter.shape[0], j:j+filter.shape[1]], filter))
                              for j in range(image.shape[1] - filter.shape[1] + 1)]
                             for i in range(image.shape[0] - filter.shape[0] + 1)]), 2)

# Function for 2x2 max pooling operation
def max_pooling(image, filter_size=(2, 2), stride=2):
    output_shape = ((image.shape[0] - filter_size[0]) // stride) + 1
    return np.array([[np.max(image[i:i+filter_size[0], j:j+filter_size[1]])
                      for j in range(0, image.shape[1], stride)]
                     for i in range(0, image.shape[0], stride)])

# Function for sigmoid activation
def sigmoid_activation(image):
    return 1 / (1 + np.exp(-image))

# Convolution and applying biases
fm1 = convolve2D(I, f1_1) + b1_1
fm2 = convolve2D(I, f1_2) + b1_2

print("Feature Map 1:")
print(fm1)

print("\nFeature Map 2:")
print(fm2)
# Apply 2x2 max pooling
pooled_fm1 = max_pooling(fm1, filter_size=(2, 2), stride=2)
pooled_fm2 = max_pooling(fm2, filter_size=(2, 2), stride=2)

print("Max pooling on Feature Map 1:")
print(pooled_fm1)

print("\nMax pooling on Feature Map 2:")
print(pooled_fm2)

# Apply sigmoid activation
final_fm1 = sigmoid_activation(pooled_fm1)
final_fm2 = sigmoid_activation(pooled_fm2)

# Print the results
print("sigmoid activation on Feature Map 1:")
print(final_fm1)

print("\nsigmoid activation on Feature Map 2:")
print(final_fm2)
