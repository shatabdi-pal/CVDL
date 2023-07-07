#Programming Assignment#1 Part#3
#Implemenation of SIFT features using OpenCv Library


import cv2 as cv
import sys

#Load the image
image_1 = cv.imread("SIFT1_img.jpg")
image_2 = cv.imread("SIFT2_img.jpg")

#error checking for loading images
if image_1 is None:
    sys.exit("Unable to read image 1")
if image_2 is None:
    sys.exit("Unable to read image 2")

# Initialize SIFT detector
sift = cv.SIFT_create()

# Create a Brute-Force matcher
bf = cv.BFMatcher()

#converting to gray scale
image_1 = cv.cvtColor(image_1, cv.COLOR_BGR2GRAY)
image_2 = cv.cvtColor(image_2, cv.COLOR_BGR2GRAY)

#detecting keypoints for image 1
key_points = sift.detect(image_1, None)


def visualize_keypoints(image, keypoints):
    # Draw keypoints on the image
    image_with_keypoints = cv.drawKeypoints(image, keypoints, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv.imshow("First Image with Keypoints", image_with_keypoints)
    cv.waitKey(0)
    cv.destroyAllWindows()

def match_key_points(image_1, image_2):
    # Detect keypoints and compute descriptors for both images
    key_points_1, descriptor_1 = sift.detectAndCompute(image_1, None)
    key_points_2, descriptor_2 = sift.detectAndCompute(image_2, None)

    # Match keypoints using k-nearest neighbors
    matches = bf.knnMatch(descriptor_1, descriptor_2, k=2)
    desired_matches = []
    for x, y in matches:
        if x.distance < 0.75 * y.distance:
            desired_matches.append(x)

    # Sort matches by distance
    desired_matches = sorted(desired_matches, key = lambda z:z.distance)

    #Compute top 10% of keypoints matches
    keypoints_retain = int(len(desired_matches) * 0.1)
    top_matches = desired_matches[:keypoints_retain]


    #visulaize the keypoints on the images
    image1_with_keypoints = cv.drawKeypoints(image_1, key_points_1, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image2_with_keypoints = cv.drawKeypoints(image_2, key_points_2, None, flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #Draw connecting lines with matched keypoints
    result_image = cv.drawMatches(image1_with_keypoints, key_points_1, image2_with_keypoints, key_points_2, top_matches, None)


    #Display images with keypoints and matches
    cv.imshow('Image 1 with keypoints',image1_with_keypoints)
    cv.imshow('Image 2 with keypoints',image2_with_keypoints)
    cv.imshow('Matched keypoints',result_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Visualize SIFT keypoints for image 1
visualize_keypoints(image_1,key_points)



# Match keypoints between image 1 and image 2
match_key_points(image_1, image_2)
