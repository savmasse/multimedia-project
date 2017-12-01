# -*- coding: utf-8 -*-

import tkinter as tk
import glob
from random import choice as random_from

class Picker:

  def __init__ ( self, cb, root='./img', types=['5x5','scrambled','jigsaw'] ):
    self.root = root
    self.cb = cb
    self.types = types

    # TKinter window
    self.win = tk.Tk()

    # TKinter variables
    self.tiles  = self.IntVar('tiles')
    self.jigsaw = self.IntVar('jigsaw')
    self.shuffled  = self.IntVar('shuffled')
    self.rotated   = self.IntVar('rotated')
    self.scrambled = self.IntVar('scrambled')
    self.x2 = self.IntVar('2x2')
    self.x3 = self.IntVar('3x3')
    self.x4 = self.IntVar('4x4')
    self.x5 = self.IntVar('5x5')
    self.choice = tk.StringVar(self.win)

    # Populate TKinter UI
    tk.Checkbutton(self.win, text="2x2", variable=self.x2, command=self.update_dropdown).grid(row=1,column=0)
    tk.Checkbutton(self.win, text="3x3", variable=self.x3, command=self.update_dropdown).grid(row=2,column=0)
    tk.Checkbutton(self.win, text="4x4", variable=self.x4, command=self.update_dropdown).grid(row=3,column=0)
    tk.Checkbutton(self.win, text="5x5", variable=self.x5, command=self.update_dropdown).grid(row=4,column=0)
    tk.Checkbutton(self.win, text="Shuffled",  variable=self.shuffled,  command=self.update_dropdown).grid(row=1,column=1)
    tk.Checkbutton(self.win, text="Rotated",   variable=self.rotated,   command=self.update_dropdown).grid(row=2,column=1)
    tk.Checkbutton(self.win, text="Scrambled", variable=self.scrambled, command=self.update_dropdown).grid(row=3,column=1)
    tk.Checkbutton(self.win, text="Tiles",  variable=self.tiles,  command=self.update_dropdown).grid(row=1,column=2)
    tk.Checkbutton(self.win, text="Jigsaw", variable=self.jigsaw, command=self.update_dropdown).grid(row=2,column=2)
    self.button_pick = tk.Button(self.win, text="Pick", command=self.take)
    self.button_pick.grid(row=5,column=0)
    self.button_random = tk.Button(self.win, text="Random", command=self.take_random) # TODO: take random
    self.button_random.grid(row=5,column=2)
    self.button_random.focus_set()

    # Show UI
    self.update_dropdown()
    tk.mainloop()

  def IntVar ( self, key, default=0):
    try:
      return tk.IntVar(self.win, key in self.types)
    except:
      return tk.IntVar(self.win, default)

  def update_dropdown ( self ):
    puzzleTypes = []
    if self.tiles.get():  puzzleTypes.append('tiles')
    if self.jigsaw.get(): puzzleTypes.append('jigsaw')
    orders = []
    if self.shuffled.get():  orders.append('shuffled')
    if self.rotated.get():   orders.append('rotated')
    if self.scrambled.get(): orders.append('scrambled')
    sizes = []
    if self.x2.get(): sizes.append('2x2')
    if self.x3.get(): sizes.append('3x3')
    if self.x4.get(): sizes.append('4x4')
    if self.x5.get(): sizes.append('5x5')
    self.paths = sorted([ path
      for type in puzzleTypes
      for order in orders
      for size in sizes
      for path in glob.glob('%s/**/%s_%s_%s_*.png' % (self.root, type, order, size), recursive=True)
    ])
    if len(self.paths):
      self.dd = tk.OptionMenu(self.win, self.choice, *self.paths)
      self.dd.grid(row=0, column=0, columnspan=3)
      self.choice.set(self.paths[0])
      self.button_pick['state'] = tk.ACTIVE
      self.button_random['state'] = tk.ACTIVE
    else:
      self.dd = tk.OptionMenu(self.win, self.choice, '')
      self.dd['state'] = tk.DISABLED
      self.button_pick['state'] = tk.DISABLED
      self.button_random['state'] = tk.DISABLED

  def take ( self ):
    self.win.destroy()
    self.cb(self.choice.get())

  def take_random ( self ):
    self.win.destroy()
    self.cb(random_from(self.paths))