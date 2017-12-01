import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('img/tiles/rotated/tiles_rotated_4x4_01.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, 3)
sobel = cv2.convertScaleAbs(sobel)
s = np.sum(sobel, axis=0)
s = s[s>30000]
plt.hist(s)



cv2.namedWindow('sobel', cv2.WINDOW_NORMAL)
cv2.imshow('sobel', sobel)
cv2.waitKey(0)
cv2.destroyAllWindows()