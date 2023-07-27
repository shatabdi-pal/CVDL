#implementation for histogram equalization
import numpy as np
import matplotlib.pyplot as plt


def histogram_equalization_4bit_gray(image):
    # Calculate the histogram
    histogram, _ = np.histogram(image, bins=16, range=[0, 15])


    # Compute the Cumulative Distribution Function (CDF)
    cdf = histogram.cumsum()


    # Normalize the CDF to the range [0, 15]

    normalized_cdf = (cdf * 15 / cdf[-1]).astype(np.uint8)

    # Use the normalized CDF to equalize the image
    equalized_image = normalized_cdf[image]

    return equalized_image, normalized_cdf


# Input 4-bit grayscale image as a 2D numpy array
image = np.array([[2, 0, 3, 4, 1, 7],
                  [15, 12, 2, 1, 0, 5],
                  [14, 11, 1, 0, 1, 3],
                  [12, 10, 0, 0, 0, 1],
                  [11, 8, 1, 0, 0, 4],
                  [7, 5, 1, 2, 3, 6]], dtype=np.uint8)

# Perform histogram equalization
equalized_image, normalized_cdf = histogram_equalization_4bit_gray(image)
# print(normalized_cdf)
final_histogram , _ = np.histogram(equalized_image, bins=16, range=[0, 15])
print(final_histogram)

# Display the original and equalized images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray', vmin=0, vmax=15)
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(equalized_image, cmap='gray', vmin=0, vmax=15)
plt.title('Equalized Image')
plt.axis('off')

plt.show()

# Display the final intensity histogram following the transformation
histogram, _ = np.histogram(image, bins=16, range=[0, 15])
plt.bar(range(16), histogram, color='b', alpha=0.5, label='Original Histogram')
plt.bar(range(16), np.histogram(equalized_image, bins=16, range=[0, 15])[0], color='r', alpha=0.5,
        label='Equalized Histogram')
plt.xlabel('Intensity Level')
plt.ylabel('Frequency')
plt.title('Final Intensity Histogram')
plt.legend()
plt.show()
