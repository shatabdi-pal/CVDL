#Programming Assignment#1 Part#1
#Implemenation of Gaussian Filter, DoG Filter, Sobel Filter

import numpy as np
from PIL import Image, ImageOps
import matplotlib.pyplot as plt



#converting image into image array
def load_image(image_path):
    image = Image.open(image_path)
    image = ImageOps.grayscale(image)
    image_array = np.array(image)
    return image_array

#convolution of gray scale image array by Gaussian Filter
def gaussian_filter(image, filter_size):
    image_height, image_width = image.shape
    if filter_size == 3:
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

    elif filter_size == 5:
        filter = np.array([[1, 4, 7, 4, 1],
                           [4, 16, 26, 16, 4],
                           [7, 26, 41, 26, 7],
                           [4, 16, 26, 16, 4],
                           [1, 4, 7, 4, 1]]) / 273

        padding = 2
        padded_image = np.pad(image, ((padding, padding), (padding, padding)), mode='constant')
        filtered_image = np.zeros_like(image)
        for i in range(image_height):
            for j in range(image_width):
                patch = padded_image[i:i + filter_size, j:j + filter_size]
                filtered_image[i, j] = np.sum(patch * filter)

    return filtered_image



#convolution of gray image array by Derivatives of Gaussian Filter
def DoG_filter(image):
    gx_filter = np.array([[1, 0, -1],
                          [2, 0, -2],
                          [1, 0, -1]])
    gy_filter = np.array([[1, 2, 1],
                          [0, 0, 0],
                          [-1, -2, -1]])
    padding = 1
    padded_image = np.pad(image, padding, mode="constant")

    gx_result = np.zeros_like(image, dtype=np.float32)
    gy_result = np.zeros_like(image, dtype=np.float32)
    image_height, image_width = image.shape
    for i in range(1, image_height + 1):
        for j in range(1, image_width + 1):
            gx_result[i -1, j - 1] = np.sum(padded_image[i - 1: i + 2, j - 1 : j + 2]* gx_filter)
            gy_result[i - 1, j - 1] = np.sum(padded_image[i - 1: i + 2, j - 1: j + 2] * gy_filter)
    return gx_result, gy_result


#Display Image from array
def display_image_gaussian_filter(original_image, filtered_image):
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(filtered_image, cmap='gray')
    plt.title('Filtered Image')
    plt.axis('off')
    plt.show()

 # Display original images, gx, gy output of DoG filter
def display_dog_filter_output(gx_output, gy_output, original_image):
    plt.subplot(1, 3, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(gx_output, cmap='gray')
    plt.title("gx output")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(gy_output, cmap='gray')
    plt.title("gy output")
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    image_path_1 = "filter1_img.jpg"
    image_path_2 = "filter2_img.jpg"
    original_image_1 = load_image(image_path_1)
    original_image_2 = load_image(image_path_2)
    filtered_image_1_3X3 = gaussian_filter(original_image_1, filter_size=3)
    filtered_image_1_5X5= gaussian_filter(original_image_1, filter_size=5)
    filtered_image_2_3X3 = gaussian_filter(original_image_2, filter_size=3)
    filtered_image_2_5X5 = gaussian_filter(original_image_2, filter_size=5)

    # Displaying output after applying convolution using Gaussain Filter
    print("output from Gaussain Filter")
    print("Convolution using 3 X 3 filter of Image 1")
    display_image_gaussian_filter(original_image_1, filtered_image_1_3X3)
    print("Convolution using 5 X 5 filter of Image 1")
    display_image_gaussian_filter(original_image_1, filtered_image_1_5X5)
    print("Convolution using 3 X 3 filter of Image 2")
    display_image_gaussian_filter(original_image_2, filtered_image_2_3X3)
    print("Convolution using 5 X 5 filter of Image 2")
    display_image_gaussian_filter(original_image_2, filtered_image_2_5X5)

    # Displaying output after applying convolution using Derivative of Gaussain Filter
    print("output of image 1 from Derivative of Gaussain Filter")
    gx_filtered_output_1, gy_filtered_output_1 = DoG_filter(original_image_1)

    #Normalizing output of Derivative of Gaussain Filter
    gx_filtered_output_1 = (gx_filtered_output_1- gx_filtered_output_1.min()) / (gx_filtered_output_1.max() - gx_filtered_output_1.min())
    gy_filtered_output_1 = (gy_filtered_output_1 - gy_filtered_output_1.min()) / (gy_filtered_output_1.max() - gy_filtered_output_1.min())
    display_dog_filter_output(gx_filtered_output_1, gy_filtered_output_1, original_image_1)
    print("output of image 2 from Derivative of Gaussain Filter")
    gx_filtered_output_2, gy_filtered_output_2 = DoG_filter(original_image_2)
    gx_filtered_output_2 = (gx_filtered_output_2 - gx_filtered_output_2.min()) / (
                gx_filtered_output_2.max() - gx_filtered_output_2.min())
    gy_filtered_output_2 = (gy_filtered_output_2 - gy_filtered_output_2.min()) / (
                gy_filtered_output_2.max() - gy_filtered_output_2.min())
    display_dog_filter_output(gx_filtered_output_2, gy_filtered_output_2, original_image_2)

    #Calculation for Sobel Filter
    gx_squared_1 = np.square(gx_filtered_output_1)
    gy_squared_1 = np.square(gy_filtered_output_1)
    sobel_filter_result_1 = np.sqrt(gx_squared_1 + gy_squared_1)
    gx_squared_2 = np.square(gx_filtered_output_2)
    gy_squared_2 = np.square(gy_filtered_output_2)
    sobel_filter_result_2 = np.sqrt(gx_squared_2 + gy_squared_2)

    #Displaying output after applying convolution using Sobel filter
    print("output of image 1 from Sobel Filter")
    display_image_gaussian_filter(original_image_1, sobel_filter_result_1)
    print("output of image 2 from Sobel Filter")
    display_image_gaussian_filter(original_image_2, sobel_filter_result_2)