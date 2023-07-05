import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


#converting image into image array
def load_image(image_path):
    image = Image.open(image_path)
    image_array = np.array(image)
    return image_array

#convolution of gray scale image array by Gaussian Filter
def gaussian_filter_gray_image(image, filter_size):
    image_height, image_width = image.shape
    filter = np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]]) / 16
    padding = 1
    padded_image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
    filtered_image = np.zeros_like(image)
    for i in range(image_height):
        for j in range(image_width):
            patch = padded_image[i:i + filter_size, j:j + filter_size]
            filtered_image[i, j] = np.sum(patch * filter)

    return filtered_image


#convolution of color image array by Gaussian Filter
def gaussian_filter_color_image(image, filter_size):
    img_height, img_width, channels = image.shape
    filter = np.array([[1, 4, 7, 4, 1],
                               [4, 16, 26, 16, 4],
                               [7, 26, 41, 26, 7],
                               [4, 16, 26, 16, 4],
                               [1, 4, 7, 4, 1]]) / 273

    padding = 2
    padded_image = np.pad(image, ((padding, padding), (padding, padding), (0,0)), mode='constant')
    filtered_image = np.zeros_like(image)

    for i in range(img_height):
        for j in range(img_width):
            for c in range(channels):
                patch = padded_image[i:i + filter_size, j:j + filter_size, c]
                filtered_image[i, j, c] = np.sum(patch * filter)

    return filtered_image


#Display Image from array
def display_image(original_image, filtered_image):
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_image)
    plt.title('Filtered Image')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    image_path_1 = "filter1_img.jpg"
    image_path_2 = "filter2_img.jpg"
    original_image_1 = load_image(image_path_1)
    original_image_2 = load_image(image_path_2)
    filtered_image_1 = gaussian_filter_gray_image(original_image_1, filter_size=3)
    filtered_image_2 = gaussian_filter_color_image(original_image_2, filter_size=5)
    print("Convolution using 3 X 3 filter")
    display_image(original_image_1, filtered_image_1)
    print("Convolution using 5 X 5 filter")
    display_image(original_image_2, filtered_image_2)

