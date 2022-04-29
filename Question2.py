import matplotlib.pyplot as plt
import cv2


def imageHistogram(img):
    histogram = plt.hist(img.ravel(), 256, [0.0, 256.0]) # found this in the documentation
    plt.show()

# y-axis = gray level intensities
# x-axis = frequency that corresponds to those respective intensities
# 1 because it is in black and white
imageFog = cv2.imread('images project1/fog.jpg', 1)
print(imageFog)
print(type(imageFog))
print(imageFog.shape) # Width, height and channel

imageHistogram(imageFog)
cv2.imshow('FogImage', imageFog)

# 2.2 Find the negative point wise transform
# usually intensity levels go from 0-(L-1) where L = 256 , the negative transform is s = L-1-r where r=initial intensity level , s = final intensity level
s = 255 - imageFog # Formula from the book pg. 174
cv2.imshow('NegativeFogImage', s)

# 2.3 Histogram of the negative image
imageHistogram(s)


cv2.waitKey(0)
cv2.destroyAllWindows()
