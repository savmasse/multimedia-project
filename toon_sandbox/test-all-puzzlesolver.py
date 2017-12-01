# -*- coding: utf-8 -*-

import glob
from puzzlesolver import Puzzle

# CONFIGURATIE
puzzle_types  = ['jigsaw', 'tiles']
puzzle_orders = ['scrambled']



solver = Puzzle()

for subpath in ['%s/%s' % (pt,po) for pt in puzzle_types for po in puzzle_orders]:
  for img_path in glob.glob('./img/%s/*.png' % subpath, recursive=True):
    solver.reset()
    solver.load(img_path)
    solver.solve()
    solver.lines()
