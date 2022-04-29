import cv2
import numpy as np

# Question 1.1
BGRImageBirds = cv2.imread("images project1/birds.jpg")
HSVImageBirds = cv2.cvtColor(BGRImageBirds, cv2.COLOR_BGR2HSV)

BGRImageStone = cv2.imread("images project1/stone.jpg")
HSVImageStone = cv2.cvtColor(BGRImageStone, cv2.COLOR_BGR2HSV)


cv2.imshow('BGRimageBirds', BGRImageBirds)
cv2.imshow('HSVimageBirds', HSVImageBirds)

cv2.imshow('BGRimageStone', BGRImageStone)
cv2.imshow('HSVimageStone', HSVImageStone)

# Question 1.2
# RGB image - intensity of I in HSI space
BGRImageBirds = np.float32(BGRImageBirds)/255 # normalize all values before using them
RImage = BGRImageBirds[:,:,2] # 2
GImage = BGRImageBirds[:,:,1] # 1
BImage = BGRImageBirds[:,:,0] # 0
HSIImageBirds = np.divide(RImage+GImage+BImage, 3)

# RGB image - intensity of V in HSV space
# V is the maximum value of the among the red, green blue colors
# HSVImageBirds = np.float32(HSVImageBirds)/255
HsvV = np.maximum(np.maximum(RImage, GImage), BImage)

cv2.imshow('HSIImageBirds', HSIImageBirds)
cv2.imshow('HSVVImageBirds', HsvV)
cv2.waitKey(0)
cv2.destroyAllWindows()