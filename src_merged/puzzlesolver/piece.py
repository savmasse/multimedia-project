# -*- coding: utf-8 -*-

import cv2
import numpy as np

def bounding_box ( img ):
  mask = img if len(img.shape) == 2 else img.any(axis=2)
  xmin, xmax = np.where(np.any(mask, axis=0))[0][[0, -1]]
  ymin, ymax = np.where(np.any(mask, axis=1))[0][[0, -1]]
  return ((ymin,ymax), (xmin,xmax))

class Piece:

  def __init__ ( self ):
    pass

  def from_contour ( self, img_puzzle, img_puzzle_gray, contour, corners ):
    # TODO: ROTEREN?
    xmin,ymin = offset = contour.min(axis=0).squeeze()
    xmax,ymax =  contour.max(axis=0).squeeze()
    self.input = img_puzzle[ymin:ymax+1, xmin:xmax+1]
    self.gray = img_puzzle_gray[ymin:ymax+1, xmin:xmax+1]
    self.mask = np.logical_or(
      np.zeros((self.input.shape[:2]), np.uint8),
      self.gray > 0 ## MISSCHIEN NIET NODIG, voegt floating islands toe
    ).astype(np.uint8)
    cv2.drawContours(self.mask, [contour], -1, 255, thickness=-1, offset=tuple(-offset))
    self.contour = contour - offset
    self.corners = corners - offset # TODO: BUG: CORNERS CONTROLEREN OP BUITEN HET VELD VALLEN!!!
    return self # voor generator expressies

  def from_slice ( self, img_slice ):
    self.input = img_slice
    self.mask = np.ones(self.input.shape[:2], dtype=np.uint8)
    self.contour = np.array([ # TODO: zelfde vorm maken als andere contours
      [[0,0]],
      [[0, self.input.shape[0]]],
      [self.input.shape[1::-1]],
      [[self.input.shape[1], 0]]
    ])
    self.corners = self.contour
    return self # voor generator expressies

  def show_color ( self, block=True, title='Piece [color]' ):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    print(self.input.dtype, self.input.shape)
    cv2.imshow(title, self.input)
    if block:
      cv2.waitKey()

  def show_mask ( self, block=True, title='Piece [mask]' ):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, self.mask)
    if block:
      cv2.waitKey()

  def show_gray ( self, block=True, title='Piece [gray]' ):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, self.gray)
    if block:
      cv2.waitKey()

  def edge_neighbours_mask ( self, edge, radius): # TODO: kan kleiner!

    # TODO (!): verwijder uitstekers/gaten   (uitsteker = telt dubbel) <-> gat telt niks!
    # TODO (!): verwijder uitstekers/gaten   (uitsteker = telt dubbel) <-> gat telt niks!
    # TODO (!): verwijder uitstekers/gaten   (uitsteker = telt dubbel) <-> gat telt niks!
    # TODO (!): verwijder uitstekers/gaten   (uitsteker = telt dubbel) <-> gat telt niks!
    # TODO (!): verwijder uitstekers/gaten   (uitsteker = telt dubbel) <-> gat telt niks!
    # TODO (!): verwijder uitstekers/gaten   (uitsteker = telt dubbel) <-> gat telt niks!
    # TODO (!): verwijder uitstekers/gaten   (uitsteker = telt dubbel) <-> gat telt niks!

    pt1 = self.corners[edge].squeeze()
    pt2 = self.corners[(edge+1)%len(self.corners)].squeeze()
    mask = np.zeros_like(self.mask)
    cv2.line(mask, tuple(pt1), tuple(pt2), 255, radius)
    if np.any(np.logical_and(pt1[0]-pt2[0],pt1[1]-pt2[1])):
      cv2.line(mask, tuple(pt1), tuple(pt2), 0, 1)
    return mask*self.mask

  """
  def rotate ( self ):
    cnt = np.pad(self.corners.squeeze(), ((0,1),(0,0)), 'wrap')
    lines = np.diff(cnt, axis=0).astype(np.float)

    p = int(max(self.input.shape))
    img = np.pad(self.input, ((p,p),(p,p),(0,0)), 'constant')

    angle = - ((np.arctan2(lines[:,0], lines[:,1])*180/np.pi+180)%90).mean()
    center = tuple(np.array(img.shape[:2])/2)
    M = cv2.getRotationMatrix2D(center, angle, 1)
    rotated = cv2.warpAffine(img, M, img.shape[:2])

    ((ymin,ymax),(xmin,xmax)) = bounding_box(rotated)
    self.rotated = rotated[ymin-1:ymax+1, xmin-1:xmax+1]
  """