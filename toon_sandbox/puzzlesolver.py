# -*- coding: utf-8 -*-

# TODO: len, trim, first horizontal

import cv2
import glob
import numpy as np
import Tkinter as tk
import random
import os.path
import itertools

from matplotlib import pyplot as plt

#https://stackoverflow.com/a/31402351
def bounding_box ( mask ):
  rows = np.any(mask, axis=1)
  cols = np.any(mask, axis=0)
  ymin, ymax = np.where(rows)[0][[0, -1]]
  xmin, xmax = np.where(cols)[0][[0, -1]]
  return (ymin, ymax+1), (xmin, xmax+1)


class Puzzle:

  def __init__ ( self, path=None ):
    self.reset()
    if path:
      self.load(path)

  def reset ( self ):
    self.input = self.pieces = None

  def load ( self, path ):
    self.input = cv2.copyMakeBorder(cv2.imread(path), 1,1,1,1, cv2.BORDER_CONSTANT, value=(0,0,0)) # zero padding voor findcontours of andere

  def solve ( self ):
    self.segment()

  def segment ( self ):
    self.pieces = []

    if self.order == 'scrambled':
      input_gray = cv2.cvtColor(self.input, cv2.COLOR_BGR2GRAY)
      contours = cv2.findContours((input_gray != 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
      for i, contour in enumerate(contours):
        self.pieces.append(Piece(self.input, contour))
    else:
      raise Exception('Unsupported puzzle order: %s' % self.order)

  def view ( self, input=True ):
    if input:
      cv2.imshow('Invoer', self.input)

  def pick ( self, **enables ):
    win = tk.Tk() # PARAMS
    def IntVar ( key, default=0):
      try:
        return tk.IntVar(win, bool(enables[key]))
      except:
        return tk.IntVar(win, default)

    tiles, jigsaw = IntVar('tiles'), IntVar('jigsaw', 1)
    shuffled, rotated, scrambled = IntVar('shuffled'), IntVar('rotated'), IntVar('scrambled', 1)
    x2, x3, x4, x5 = IntVar('2x2'), IntVar('3x3'), IntVar('4x4'), IntVar('5x5', 1)
    choice = tk.StringVar(win)

    def take ( path ):
      print('[Puzzle] Loading %s' % path)
      win.destroy()
      self.reset()
      self.puzzleType, self.order, size, index = os.path.basename(path).split('_')
      self.size = int(size[0])
      self.load(path)

    def update_dropdown ():
      puzzleTypes = []
      if tiles.get():  puzzleTypes.append('tiles')
      if jigsaw.get(): puzzleTypes.append('jigsaw')
      orders = []
      if shuffled.get():  orders.append('shuffled')
      if rotated.get():   orders.append('rotated')
      if scrambled.get(): orders.append('scrambled')
      sizes = []
      if x2.get(): sizes.append('2x2')
      if x3.get(): sizes.append('3x3')
      if x4.get(): sizes.append('4x4')
      if x5.get(): sizes.append('5x5')
      update_dropdown.paths = paths = sorted([ path
        for type in puzzleTypes
        for order in orders
        for size in sizes
        for path in glob.glob('./img/**/%s_%s_%s_*.png' % (type, order, size), recursive=True)
      ])
      if len(paths):
        update_dropdown.dd = tk.OptionMenu(win, choice, *paths)
        update_dropdown.dd.grid(row=0, column=0, columnspan=3)
        choice.set(paths[0])
        update_dropdown.pick['state'] = tk.ACTIVE
        update_dropdown.random['state'] = tk.ACTIVE
      else:
        update_dropdown.dd = tk.OptionMenu(win, choice, '')
        update_dropdown.dd['state'] = tk.DISABLED
        update_dropdown.pick['state'] = tk.DISABLED
        update_dropdown.random['state'] = tk.DISABLED

    tk.Checkbutton(win, text="2x2", variable=x2, command=update_dropdown).grid(row=1,column=0)
    tk.Checkbutton(win, text="3x3", variable=x3, command=update_dropdown).grid(row=2,column=0)
    tk.Checkbutton(win, text="4x4", variable=x4, command=update_dropdown).grid(row=3,column=0)
    tk.Checkbutton(win, text="5x5", variable=x5, command=update_dropdown).grid(row=4,column=0)
    tk.Checkbutton(win, text="Shuffled",  variable=shuffled,  command=update_dropdown).grid(row=1,column=1)
    tk.Checkbutton(win, text="Rotated",   variable=rotated,   command=update_dropdown).grid(row=2,column=1)
    tk.Checkbutton(win, text="Scrambled", variable=scrambled, command=update_dropdown).grid(row=3,column=1)
    tk.Checkbutton(win, text="Tiles",  variable=tiles,  command=update_dropdown).grid(row=1,column=2)
    tk.Checkbutton(win, text="Jigsaw", variable=jigsaw, command=update_dropdown).grid(row=2,column=2)
    update_dropdown.pick = tk.Button(win, text="Pick", command=lambda:take(choice.get()))
    update_dropdown.pick.grid(row=5,column=0)
    update_dropdown.random = tk.Button(win, text="Random", command=lambda:take(random.choice(update_dropdown.paths)))
    update_dropdown.random.grid(row=5,column=2)
    update_dropdown.random.focus_set()
    update_dropdown()
    tk.mainloop()

class Piece:

  def __init__ ( self, input, contour ):
    # Bereken bounding box met 1 px padding
    offset = (-contour[:,0,0].min()+1, -contour[:,:,1].min()+1)
    shape = (contour[:,0,1].ptp()+3, contour[:,0,0].ptp()+3)

    # Verkrijg een slice view uit de originele afbeelding
    self.input = input[
      -offset[1] : -offset[1]+shape[0],
      -offset[0] : -offset[0]+shape[1]
    ]

    # Verwijder de offset uit de contour data
    contour[:,0,0] += offset[0]
    contour[:,0,1] += offset[1]
    self.contour = contour

    # Genereer het masker met 1px zero padding o.b.v. de contour info
    self.mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(self.mask, [contour], 0, 255, thickness=-1)

  def lines ( self, method='simple' ):

    # VERWIJDER MIJ NA TESTEN (hergebruik contour uit self)
    if method == 'simple':
      cnt = cv2.findContours(self.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1][0].squeeze()[:,None,:]
      s = np.hstack((cnt, np.roll(cnt,-1,0)))
    else:
      cnt = cv2.findContours(self.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1][0].squeeze()[:,None,:]
      roll = np.argmin(cnt[::-1,1] == cnt[0,1])
      s = np.dstack((np.roll(cnt,roll,0), np.roll(cnt,roll-1,0)))

    d = np.diff(s, axis=1)
    o = np.apply_along_axis(gridAngle, 2, d)
    h = np.hstack((o<2, (o%3).astype(np.bool)))

    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    demo = (self.mask[:,:,np.newaxis].repeat(3,2)/4).astype(np.uint8)
    lines = sorted([ # HOUDT ORI BIJ VAN MINSTE LIJNEN, MAAR ZAL PROBLEMEN GEVEN BIJ RECHTOPSTAANDE (orthogonale) afbeeldingen
      [
        np.vstack(seg[None,:,:] for _,seg in line)
        for _,line in itertools.groupby(zip(h,s), key=lambda z:z[0])
      ]
      for h in np.rollaxis(h,1)
    ], key=lambda arr:len(arr))[0]

    for line in lines:
      diff = np.diff(line, axis=1) # D wordt hier herberekend...
      length = np.hypot(diff[:,:,0],diff[:,:,1]).sum()
      if length > 15:
        print(length)
        color = (np.random.randint(0,255),np.random.randint(0,255),np.random.randint(0,255))
        for seg in line:
          cv2.line(demo, tuple(seg[0]), tuple(seg[1]), color, 1, 4)
        cv2.imshow('output', demo)
        cv2.waitKey(0)

def gridAngle ( d ):
  y,x = d
  if x==y:
    return 3
  else:
    if x == 0:
      return 0
    elif y == 0:
      return 2
    else:
      return 1



# TESTING/DEBUGGING
no_dialog = True

if __name__ == '__main__':
  puzzle = Puzzle()
  if no_dialog:
    puzzle.pick()
  else:
    puzzle.type  = 'jigsaw'
    puzzle.order = 'scrambled'
    puzzle.size  = 5
    puzzle.load('./img/jigsaw/scrambled/jigsaw_scrambled_5x5_00.png')
  puzzle.solve()
  for piece in puzzle.pieces:
    piece.lines()
  cv2.destroyAllWindows()