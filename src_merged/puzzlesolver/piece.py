# -*- coding: utf-8 -*-

###############################################################################

blue  = (255,0,0)
green = (0,255,0)
red   = (0,0,255)

###############################################################################

from .exception import Unsolvable
from snippets   import draw_cross, show, seg_intersect

import cv2
import numpy as np
from numpy import array as arr
from matplotlib import pyplot as plt

class Piece:

  def __init__ ( self ):
    pass

  def from_contour ( self, img_puzzle, img_puzzle_gray, contour, corners, jigsaw=False ):
    xmin,ymin = offset = contour.min(axis=0).squeeze()
    xmax,ymax =  contour.max(axis=0).squeeze()
    self.offset = offset
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

  def from_slice ( self, img_slice, offset ):
    self.input = img_slice
    self.offset = offset
    self.mask = np.ones(self.input.shape[:2], dtype=np.uint8)
    self.contour = (arr([
      [[0,0]],
      [[0,1]],
      [[1,1]],
      [[1,0]]
    ]) * arr(self.input.shape[:2])[::-1]-1).astype(int)
    self.corners = self.contour
    self.tips = False

    return self # voor generator expressies

  def edge_neighbours_mask ( self, side, radius ):
    mask = np.zeros_like(self.mask)

    # Draw rectangle side
    pt1 = self.corners[side].squeeze()
    pt2 = self.corners[(side+1)%len(self.corners)].squeeze()
    cv2.line(mask, tuple(pt1), tuple(pt2), 1, radius, lineType=8)

    # Erase tip (draw black square orthogonal on the tip
    if self.tips and self.tips[side]:
      _,_,(l,r),_,_,_,m = self.tips[side]
      o = 2*m-l-r
      lo = l+o
      ro = r+o
      pts = np.array([l, lo, ro, r], dtype=int)
      cv2.fillConvexPoly(mask, pts, 0)

    # Dont mask outer border pixels to avoid picking up background colors from aliasing
    if np.any(np.logical_and(pt1[0]-pt2[0],pt1[1]-pt2[1])):  # weinig invloed..
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
        cx,cy = c = top + o
        dx,dy = d = top - o
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
        pr = seg_intersect(a,b, right,or2).astype(int)

        # Distance from corner A to the middle of the Tip
        # (already calculated, but not sooo heavy to recalculate)
        m = (pl+pr)/2
        am = m-a
        dam = np.hypot(*am)

        self.tips.append((
          signed_dists[i_top] < 0, # TRUE = MALE; FALSE = FEMALE
          top,
          (left, right),
          (pl, pr), # index in tips niet meer wijzigen!
          arc[:,None,:], # CONTOUR (open)
          dam/ds,
          m# relative position of the top [0,1] where 0 = @A and 1=@B
        ))

        #SELFTEST
        if diff == 0:
          print('Warning: guessed which side is left or right for side %i' % side)
          self.show_info()
      else:
        self.tips.append(None)

    if not len([True for tip in self.tips if tip is not None]):
      self.tips = False
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

  def show_info ( self, out=None, block=True, title='Piece [info]', view=True, offset=[0,0], stacked=True, id=None ):
    if out is None:
      out = cv2.copyMakeBorder(64*np.repeat(self.mask[:,:,None], 3, axis=2), *offset, *offset, cv2.BORDER_CONSTANT, value=black)
      offset = [10,10]

    # Draw ID in centroid of the contour
    if id is not None:
      txtsize = cv2.getTextSize(str(id), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
      M = cv2.moments(self.corners)
      center = arr([
        M['m10']/M['m00'] - txtsize[0]/2,
        M['m01']/M['m00'] + txtsize[1]/2
      ])
      cv2.putText(out, str(id), tuple((offset+center).astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 1, (64,64,64), 2)

    # Draw contours
    cv2.drawContours(out, [self.contour], -1, (205,72,72), 2, offset=tuple(offset))

    # Draw fitted rectangle
    # TODO: * mask * ~mask_tip
    for i in range(len(self.corners)):
      length = self.sides[i]
      nextlength = self.sides[(i+1)%len(self.sides)]

      if length == nextlength: # square
        color = (100,16,16)
      elif length < nextlength: # short
        color = (200,0,200)
      else: # long
        color = (0,200,200)

      pts = offset+np.roll(self.corners, -i, axis=0)[:2, 0]
      pt1 = pts[0]
      pt2 = pts[1]
      cv2.line(out, tuple(pt1), tuple(pt2), color, 1)

    # Draw tips
    if self.tips != False:
      for tip in self.tips:
        if tip is None:
          continue

        male, top, (left,right), (proj_l,proj_r), arc, rel_dist,_ = tip

        # Contour
        cv2.polylines(out, [arc+offset], False, green, 2, lineType=8)

        # Markers
        top_color = (255,0,255) if male else red
        draw_cross(out, offset+top, 5, top_color)
        draw_cross(out, offset+left,  5, green)
        draw_cross(out, offset+right, 5, blue)
        draw_cross(out, offset+proj_l, 3, green)
        draw_cross(out, offset+proj_r, 3, blue)

        # Ratio
        cv2.putText(out,'%.1f'%(100*rel_dist), tuple(offset+[-12,6]+3*top-left-right), cv2.FONT_HERSHEY_SIMPLEX, .33, (255,255,255), 1)

    # Show on screen
    if stacked:
      img = np.hstack((cv2.copyMakeBorder(self.input, *offset, *offset, cv2.BORDER_CONSTANT, value=(0,0,0)), out))
      if view:
        show(img, block, title)

    return out

  def draw_contour ( self, out, color=255, **line_params):
    cv2.drawContours(out, [self.contour], -1, color=color, **line_params)

  def draw_side ( self, out, side, color=255, offset=[0,0], **line_params):
    pt1 = offset + self.corners[side, 0]
    pt2 = offset + self.corners[(side+1)%len(self.corners), 0]
    cv2.line(out, tuple(pt1), tuple(pt2), color, **line_params)