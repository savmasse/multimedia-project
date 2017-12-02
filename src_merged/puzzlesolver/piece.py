# -*- coding: utf-8 -*-

from .exception import Unsolvable, Guess
from snippets   import draw_cross, show, seg_intersect
from colors     import red,green,blue,black,white

import cv2
import numpy as np
from matplotlib import pyplot as plt

class Piece:

  def __init__ ( self ):
    pass

  def from_contour ( self, img_puzzle, img_puzzle_gray, contour, corners, jigsaw=False ):
    xmin,ymin = offset = contour.min(axis=0).squeeze()
    xmax,ymax =  contour.max(axis=0).squeeze()
    self.input = img_puzzle[ymin:ymax+1, xmin:xmax+1]
    self.gray = img_puzzle_gray[ymin:ymax+1, xmin:xmax+1]
    self.mask = np.logical_or(
      np.zeros((self.input.shape[:2]), np.uint8),
      self.gray > 0 # voegt floating islands toe
    ).astype(np.uint8)
    cv2.drawContours(self.mask, [contour], -1, 1, thickness=-1, offset=tuple(-offset))
    self.contour = contour - offset
    self.corners = corners - offset

    if jigsaw:
      self.detect_tips()
    else:
      self.tips = False

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
    self.tips = False

    return self # voor generator expressies

  def edge_neighbours_mask ( self, side, radius ):
    mask = np.zeros_like(self.mask)

    # Draw rectangle side
    pt1 = self.corners[side].squeeze()
    pt2 = self.corners[(side+1)%len(self.corners)].squeeze()
    cv2.line(mask, tuple(pt1), tuple(pt2), 1, radius, lineType=8)

    # Erase tip
    if self.tips and self.tips[side]:
      _,_,_,(pl,pr),_ = self.tips[side]
      cv2.line(mask, tuple(pl), tuple(pr), 0, radius+5, lineType=8)

    # Dont mask outer border pixels to avoid picking up background colors from aliasing
    if np.any(np.logical_and(pt1[0]-pt2[0],pt1[1]-pt2[1])):  # indien schuin
      cv2.drawContours(mask, [self.contour], -1, 0, 1, 8)

    return mask*self.mask

  def detect_tips ( self, threshold=4 ):
    corners = self.corners.squeeze()
    contour = self.contour.squeeze()

    idx_corners= [np.argmax(np.all(contour == corner, axis=1)) for corner in corners] # [!] np.all != efficiënt

    self.tips = []
    for side in range(len(corners)):
      start = idx_corners[side]+1
      stop  = idx_corners[(side+1)%len(corners)]

      # Distance from edge<e> to line <a,b>
      e = np.roll(contour, -start, axis=0)[:stop-start]
      ex = e[:,0]
      ey = e[:,1]
      ax,ay = a = corners[side]
      bx,by = b = corners[(side+1)%len(corners)]
      sx, sy = s = b-a
      ds = np.hypot(sx,sy)
      signed_dists = ((sy*ex - sx*ey) + (bx*ay-by*ax))/ds
      dists = np.abs(signed_dists)


      if len(dists) and dists.max() > threshold: # [!] AFWIJKINGEN > 3 BESTAAN => !!! problemen bij roteren # TODO: FIX
        i_top = np.argmax(dists)
        d_top = dists[i_top]
        top = e[i_top]
        arc = np.array([pt for dist,pt in zip(dists,e) if dist > threshold])

        # Distance from orthogonal line<o> through top to arc
        # > d_top + 10: zonder +10 is wrs voldoende, mr om zeker te spelen
        arcx = arc[:,0]
        arcy = arc[:,1]
        ox,oy = o  = ([1,-1]*s[::-1]*(d_top+10)/ds).astype(np.int) # orthogonaal
        cx,cy = top + o
        dx,dy = top - o
        signed_dists_ortho = ((oy*arcx - ox*arcy) + (dx*cy-dy*cx))/np.hypot(ox,oy)
        dists_ortho = np.abs(signed_dists_ortho)
        # > determine left or right based on manhattan distance to first corner
        p1,p2 = arc[[np.argmin(dists_ortho), np.argmax(dists_ortho)]]
        diff = np.abs(p1-a).sum() - np.abs(p2-a).sum()
        if diff > 0:
          left, right = p1,p2
        else:
          left, right = p2,p1

        # Projection of left<pl> and right<pr> extrema to fitted rectangle contour
        ol2 = left + o
        or2 = right + o
        pl = seg_intersect(a,b, left,ol2).astype(int)
        pr= seg_intersect(a,b, right,or2).astype(int)


        self.tips.append((
          signed_dists[i_top] < 0, # TRUE = MALE; FALSE = FEMALE
          top,
          (left, right),
          (pl, pr), # positie niet wijzigen
          arc[:,None,:] # CONTOUR (open),
        ))

        #SELFTEST
        if diff == 0:
          print('Warning: guessed which side is left or right for side %i' % side)
          self.show_info()
      else:
        self.tips.append(None)

    if not len([True for tip in self.tips if tip is not None]):
      self.tips = False
      #self.show_info()
      raise Unsolvable('Jigsaw zonder tips')


  def show_color ( self, block=True, title='Piece [color]', view=True ):
    if view: show(self.input, block, title)
    return self.input

  def show_mask ( self, block=True, title='Piece [mask]', view=True ):
    if view: show(self.mask, block, title)
    return self.mask

  def show_gray ( self, block=True, title='Piece [gray]', view=True ):
    if view: show(self.gray, block, title)
    return self.gray

  def show_info ( self, block=True, title='Piece [info]', view=True ):
    offset = [10,10]
    img = cv2.copyMakeBorder(64*np.repeat(self.mask[:,:,None], 3, axis=2), *offset, *offset, cv2.BORDER_CONSTANT, value=black)

    # Draw contours
    cv2.drawContours(img, [self.contour], -1, (205,72,72), 2, offset=tuple(offset))

    # Draw fitted rectangle
    # TODO: * mask * ~mask_tip
    for i in range(len(self.corners)):
      if self.sides[i] == 0: # square
        color = (100,16,16)
      elif self.sides[i] == 1: # short
        color = (200,0,200)
      elif self.sides[i] == 2: # long
        color = (0,200,200)
      else:
        color = (70,96,162)
      pts = offset+np.roll(self.corners, -i, axis=0)[:2, 0]
      pt1 = pts[0]
      pt2 = pts[1]
      cv2.line(img, tuple(pt1), tuple(pt2), color, 1)

    # Draw tips
    if self.tips != False:
      for tip in self.tips:
        if tip is None:
          continue

        male, top, (left,right), (proj_l,proj_r), arc = tip

        # Tip contour
        cv2.polylines(img, [arc+offset], False, green, 2, lineType=8)

        # Tip markers
        top_color = (255,0,255) if male else red
        draw_cross(img, offset+top, 5, top_color)
        draw_cross(img, offset+left,  5, green)
        draw_cross(img, offset+right, 5, blue)
        draw_cross(img, offset+proj_l, 3, green)
        draw_cross(img, offset+proj_r, 3, blue)

    # Show on screen
    if view:
      img = np.hstack((cv2.copyMakeBorder(self.input, *offset, *offset, cv2.BORDER_CONSTANT, value=(0,0,0)), img))
      show(img, block, title)

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