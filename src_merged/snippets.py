# -*- coding: utf-8 -*-

import numpy as np
import cv2

def bounding_box ( img ):
  mask = img if len(img.shape) == 2 else img.any(axis=2)
  xmin, xmax = np.where(np.any(mask, axis=0))[0][[0, -1]]
  ymin, ymax = np.where(np.any(mask, axis=1))[0][[0, -1]]
  return ((ymin,ymax), (xmin,xmax))

def show ( img, block, title ):
  cv2.namedWindow(title, cv2.WINDOW_NORMAL)
  cv2.imshow(title, img)
  if block:
    cv2.waitKey()

def draw_cross ( img, center, radius, color, h=True, v=True ):
  center = np.array(center)
  if v:
    cv2.line(img, tuple(center-[radius,0]), tuple(center+[radius,0]), color, 1, lineType=8)
  if h:
    cv2.line(img, tuple(center-[0,radius]), tuple(center+[0,radius]), color, 1, lineType=8)

# aanpassing van https://stackoverflow.com/a/3252222/2646357
def seg_intersect(a1,a2, b1,b2) :
    da = a2-a1
    db = b2-b1
    dp = a1-b1
    dap = da[::-1]*[-1,1]
    denom = np.dot( dap, db)
    num = np.dot( dap, dp )
    return (num / denom.astype(float))*db + b1