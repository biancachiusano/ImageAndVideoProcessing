import cv2
import numpy as np

# Question 1.1
# Reading the images and  transforming to HSV color space
BGRImageBirds = cv2.imread("images project1/birds.jpg")
HSVImageBirds = cv2.cvtColor(BGRImageBirds, cv2.COLOR_BGR2HSV)

BGRImageStone = cv2.imread("images project1/stone.jpg")
HSVImageStone = cv2.cvtColor(BGRImageStone, cv2.COLOR_BGR2HSV)

cv2.imshow('OriginalBirds', BGRImageBirds)
cv2.imshow('HSVimageBirds', HSVImageBirds)
cv2.imshow('OriginalStone', BGRImageStone)
cv2.imshow('HSVimageStone', HSVImageStone)

# Question 1.2
# RGB image - intensity of I in HSI space

BGRImageBirds = np.float32(BGRImageBirds)/255 # normalize all values before using them
# Getting the single RGB values
RImage = BGRImageBirds[:,:,2]
GImage = BGRImageBirds[:,:,1]
BImage = BGRImageBirds[:,:,0]
# Adding the single values and getting the average by dividing by 3
HSIImageBirds = np.divide(RImage+GImage+BImage, 3)

# RGB image - intensity of V in HSV space
# V is the maximum value of the among the red, green blue colors
HsvVBirds = np.maximum(np.maximum(RImage, GImage), BImage)
cv2.imshow('HSIImageBirds', HSIImageBirds)
cv2.imshow('HSVVImageBirds', HsvVBirds)
cv2.waitKey(0)
cv2.destroyAllWindows()

# This is the same code as above but for the Stone image
'''
BGRImageStone = np.float32(BGRImageStone)/255 
RImage = BGRImageStone[:,:,2] # 2
GImage = BGRImageStone[:,:,1] # 1
BImage = BGRImageStone[:,:,0] # 0
HSIImageStone = np.divide(RImage+GImage+BImage, 3)
HsvVStone = np.maximum(np.maximum(RImage, GImage), BImage)
cv2.imshow('HSIImageStone', HSIImageStone)
cv2.imshow('HSVVImageStone', HsvVStone)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''




