import cv2
import sys

# read the images
input_image_1 = cv2.imread('SIFT1_img.jpg')
input_image_2 = cv2.imread('SIFT2_img.jpg')

# error checking for loading images
if input_image_1 is None:
    sys.exit("Unable to read image 1")
if input_image_2 is None:
    sys.exit("Unable to read image 2")


# resizing images
desired_width = 800
desired_height = 800

# Resize the image
image_1 = cv2.resize(input_image_1, (desired_width, desired_height))
image_2 = cv2.resize(input_image_2, (desired_width, desired_height))

# create SIFT object
sift = cv2.SIFT_create()

# detect SIFT features in both images
keypoints_1, descriptors_1 = sift.detectAndCompute(image_1, None)
keypoints_2, descriptors_2 = sift.detectAndCompute(image_2, None)

# create feature matcher
bf = cv2.BFMatcher()
# Match keypoints using k-nearest neighbors
matches = bf.knnMatch(descriptors_1, descriptors_2, k=2)

desired_matches = []
for x, y in matches:
    if x.distance < 0.75 * y.distance:
        desired_matches.append(x)

# sort matches by distance
desired_matches = sorted(desired_matches, key=lambda x: x.distance)

# Compute top 10% of keypoints matches
keypoints_retain = int(len(desired_matches) * 0.1)
top_matches = desired_matches[:keypoints_retain]

# visulaize the keypoints on the images

image1_with_keypoints = cv2.drawKeypoints(image_1, keypoints_1, None)
image2_with_keypoints = cv2.drawKeypoints(image_2, keypoints_2, None)

# Draw connecting lines with matched keypoints
matched_img = cv2.drawMatches(image_1, keypoints_1, image_2, keypoints_2, top_matches, None)
# show the image

cv2.imshow('Image 1 with keypoints', image1_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Image 2 with keypoints', image2_with_keypoints)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imshow('Result image with matched keypoints', matched_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
