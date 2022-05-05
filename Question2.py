import matplotlib.pyplot as plt
import cv2
import numpy as np

# Task 2
def imageHistogram(img):
    # img.ravel because it takes in a 1-d array
    histogram = plt.hist(img.ravel(), 256, [0.0, 256.0]) # 256 bins, and range[0-256]
    plt.show()

# y-axis = gray level intensities
# x-axis = frequency that corresponds to those respective intensities
# Reading the image, and the other parameter is 1 because it is in black and white
imageFog = cv2.imread('images project1/shadows.jpg', 1)
print(imageFog)
print(type(imageFog))
print(imageFog.shape) # Width, height and channel

imageHistogram(imageFog)
cv2.imshow('FogImage', imageFog)

# 2.2 Find the negative point wise transform
# usually intensity levels go from 0-(L-1) where L = 256 , the negative transform is s = L-1-r where r=initial intensity level , s = final intensity level
s = 255 - imageFog # Formula from the book pg. 174
cv2.imshow('NegativeFogImage', s)
cv2.imwrite('NegativeShadowImage.jpg', s)

# 2.3 Histogram of the negative image
imageHistogram(s)


# 2.4 Power law pointwise transform
# Read the image
imageFog = cv2.imread('images project1/shadows.jpg', 1)
gamma = [0.1,0.5,1.5,2.0] # testing these 4 gamma values

for x in gamma:
    # x is the gamma value in this case (the exponent)
    correctedImage = np.array(255*(imageFog/255) ** x, dtype='uint8')
    cv2.imshow('transformedShadow'+str(x) , correctedImage)


cv2.waitKey(0)
cv2.destroyAllWindows()
