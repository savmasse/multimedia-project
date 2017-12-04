# -*- coding: utf-8 -*-

from timeit import default_timer as now

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

class Time:
  def __init__ ( self, title, log=True, msg='%s: %.03fs' ):
    self.title = title
    self.log = log
    self.msg= msg

  def __enter__ ( self ):
    self.t0 = now()
    return self

  def __exit__(self, *args):
    self.dt = now() - self.t0

    if self.log:
      self.print()

  def pause ( self ):
    class Stop:
      def __init__( self, timer ):
        self.timer = timer
      def __enter__( self ):
        self.t0 = now()
        return self
      def __exit__( self, *args ):
        self.timer.t0 += now()-self.t0

    return Stop(self)


  def print ( self ):
    print(self.msg % (self.title, self.dt))

class Count:
  def __init__ ( self, title, init=[], log=True, msg='%s: %i'):
    self.title = title
    self.log = log
    self.msg = msg
    self.list = init

  def __enter__ ( self ):
    return self

  def __exit__(self, *args):
    if self.log:
      self.print()

  def count ( self, n=1 ):
    self.list.append(n)

  def print ( self ):
    print(self.msg % (self.title, sum(self.list)))


class Grid_solution_stats:

  def __init__ ( self, grid, title, show=True ):
    self.title = title
    self.grid = grid
    self.n0 = self.grid.tot_n_candidates
    self.show = show

  def __enter__ ( self ):
    self.t0 = now()

  def __exit__ ( self, *args ):
    if self.show:
      print('Grid: %s\t %sms → %i removed ' % (
        self.title,
        str(int(1000*(now()-self.t0))).rjust(3),
        self.n0 - self.grid.tot_n_candidates
      ))
