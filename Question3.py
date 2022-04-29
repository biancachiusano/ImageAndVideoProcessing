import cv2
import numpy as np

# Special effects
# 3.1 Convert image into polar coordinates

# http://amroamroamro.github.io/mexopencv/matlab/cv.linearPolar.html
# LinearPolar (image, center, maximumradius)
img = cv2.imread('images project1/yellow.jpeg')
W = img.shape[0]
H = img.shape[1]
cent_x = W/2
cent_y = H/2
maxRadius = np.sqrt(W**2 + H**2)/2
img2 = cv2.linearPolar(img, (cent_x, cent_y), maxRadius, cv2.WARP_FILL_OUTLIERS)

cv2.imshow('before', img)
cv2.imshow('polar', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3.2