import cv2
import numpy as np
from matplotlib import pyplot as plt

def cb (_):
  blocksize = cv2.getTrackbarPos('blocksize', cb.win)
  ksize = cv2.getTrackbarPos('ksize', cb.win)
  k = cv2.getTrackbarPos('k', cb.win)
  th = cv2.getTrackbarPos('th', cb.win)
  dst = cv2.cornerHarris(cb.mask, blocksize, ksize, k/1000)
  dst = np.abs(dst)
  dst = cv2.convertScaleAbs(dst, alpha=255/dst.max())
  dst = dst*(dst>th)
  edges = cv2.Canny(cb.mask_gray,50,150,apertureSize = 3)
  cv2.imshow(cb.win, edges)
  lines = cv2.HoughLines(edges,1,3*np.pi/180,51)
  print(len(lines))
  out = cb.img.copy()
  for rho,theta in lines[:,0,:]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 2000*(-b))
    y1 = int(y0 + 2000*(a))
    x2 = int(x0 - 2000*(-b))
    y2 = int(y0 - 2000*(a))
    cv2.line(out,(x1,y1),(x2,y2),(0,0,255),2)
  cv2.imshow(cb.win, out)

img = cv2.imread('img/jigsaw/scrambled/jigsaw_scrambled_5x5_02.png')
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = (img_gray > 0).astype(np.float32)
#mask = cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5,5), dtype=np.uint8))

cb.img = img
cb.win = 'mask'
cb.mask = mask
cb.mask_gray = mask.astype(np.uint8)*255
cv2.namedWindow(cb.win, cv2.WINDOW_NORMAL)
cv2.createTrackbar('blocksize', cb.win, 1, 20, cb)
cv2.createTrackbar('ksize', cb.win, 1, 20, cb)
cv2.createTrackbar('k', cb.win, 50, 1000, cb)
cv2.createTrackbar('th', cb.win, 0, 255, cb)
cb(None)
cv2.waitKey(0)