import cv2
import numpy as np
from matplotlib import pyplot as plt

# Translate the image vertically by 1/3 of the image size
# Reading the image
img = cv2.imread("images project1/birds.jpg")
# Getting the height and width of the image
height, width = img.shape[:2]
# only translating vertically by 1/3 of the image size, so the translation for the x-direction is 0
translationMat = np.float32([[1,0,0],[0, 1, height/3]])
# Using the warpAffine function
translation = cv2.warpAffine(img, translationMat, (width, height))
cv2.imshow('Original', img)
cv2.imshow('Translated', translation)


# FFT, FFT magnitude
# Method takes in the input image
def fourierTransform(imgInput):
    f = np.fft.fft2(imgInput)
    fshift = np.fft.fftshift(f)
    # Abs for the magnitude because fshift is complex
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    plt.imshow(magnitude_spectrum, cmap = 'gray')
    plt.show()

img2 = cv2.imread('images project1/birds.jpg', 0)
fourierTransform(img2)
# Also turning the translated image to a grayscale image
grayTranslated = cv2.cvtColor(translation, cv2.COLOR_RGB2GRAY)
fourierTransform(grayTranslated)

cv2.waitKey(0)
cv2.destroyAllWindows()

