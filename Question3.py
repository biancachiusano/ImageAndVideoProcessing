import cv2
import numpy as np
from scipy import ndimage
# Special effects
# 3.1 Convert image into polar coordinates

# read the image
img = cv2.imread('images project1/yellow.jpeg')
W = img.shape[0] # width - rows of the matrix
H = img.shape[1] # height - col of the matrix
cent_x = W/2 # find the center by dividing by 2
cent_y = H/2
maxRadius = np.sqrt(W**2 + H**2)/2 #This is the formula to find the radius
img2 = cv2.linearPolar(img, (cent_x, cent_y), maxRadius, cv2.WARP_FILL_OUTLIERS) # LinearPolar (image, center, maximumradius)

cv2.imshow('before', img)
cv2.imshow('polar', img2)
img2 = ndimage.rotate(img2, 90) # Rotating to see the results better


# 3.2
# read the image
imgBeforeCartoon = cv2.imread('images project1/flower.jpeg')
cv2.imshow('Before', imgBeforeCartoon)

# Perform color quantization technique
# takes in the input image and the number of cluster k
def cartoonize(k):

    # input data for clustering
    # Reshaping to make one column = row * col of the input image
    reshaped = np.float32(imgBeforeCartoon).reshape((-1, 3))
    # color quantization is done in a single column --> reshape function reshapes the input data to a single column
    # defining the criteria -> iterates until the criteria is met
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    # applying cv2.kmeans function --> performs the K-means clustering problem
    _, label, center = cv2.kmeans(reshaped, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    # Reshape the output data to size of the input image
    result = center[label.flatten()]
    result = result.reshape(imgBeforeCartoon.shape) # reshape to match the input image
    cv2.imshow("result", result)

    # find edges in the original image
    # Convert the input image to gray scale
    grayImage = cv2.cvtColor(imgBeforeCartoon, cv2.COLOR_BGR2GRAY)  # changes the color space of an image -> convert to grayscale
    # Perform adaptive threshold function, Threshold in this case is 5. All pixels with a lower threshold will be left in the background
    edges = cv2.adaptiveThreshold(grayImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)  # finds darker edges and returns a mask of the binary image
    cv2.imshow('edges', edges)


    # Combine edges and quantized result
    # first smooth the result with medianBlur function
    blurred = cv2.medianBlur(result, 3)
    # Combine the result and edges to get final cartoon effect
    cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
    cv2.imshow('cartoon', cartoon)

# get the output of 8 centers containing RGB value
# out new image will only contain 8 unique colors
cartoonize(8)


cv2.waitKey(0)
cv2.destroyAllWindows()