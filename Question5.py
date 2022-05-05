import cv2
import random

# 5.1

# adding pepper and salt noise
def add_noise(img):
    r, c = img.shape
    # Performing this 10000 times so for 10000 pixels
    for i in range(10001):
        # randomly choosing a pixel to make white
        y_white = random.randint(0, r-1)
        x_white = random.randint(0, c-1)
        img[y_white][x_white] = 255
        # randomly choosing a pixel to make black
        y_black = random.randint(0, r-1)
        x_black = random.randint(0, c-1)
        img[y_black][x_black] = 0
    return img

# reading the image and making it grayscale in order to see the noise
img = cv2.imread('images project1/pink.jpg', cv2.IMREAD_GRAYSCALE)
noisyImage = add_noise(img)
cv2.imshow('grayscale Original', img)
cv2.imshow('With_Noise', noisyImage)


# 5.3 Denoise the image, as the last two paramenters increase the more denoised the image is
denoised = cv2.fastNlMeansDenoising(noisyImage, None, 20, 20)
cv2.imwrite('Denoised.jpg', denoised)

cv2.waitKey(0)
cv2.destroyAllWindows()