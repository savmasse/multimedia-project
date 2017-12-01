# -*- coding: utf-8 -*-

import glob
import time
import puzzlesolver as ps
import numpy as np

def execute ():
  # Kies welke puzzels je wil laden
  paths = []
  #paths.extend(glob.glob('../img/**/tile*rotated*.png', recursive=True))
  #paths.extend(glob.glob('../img/**/tile*shuffled*.png', recursive=True))
  #paths.extend(glob.glob('../img/**/*jigsaw*scrambled*.png', recursive=True))
  paths.extend(glob.glob('../img/**/*.png', recursive=True))

  times = []
  t0 = time.time()
  for path in sorted(paths):
    puzzle = ps.Puzzle(path)
    try:
      ta = time.time()
      puzzle.extract_pieces()
      dt = time.time()-ta
      times.append(dt)

      print(puzzle.correct(path), '%.02f' % np.round(1000*(dt), 2), path)
      puzzle.solve()
    except ps.Unsolvable as exc:
      print(puzzle.correct(path), 'ERR ', path)
      print(exc)
      continue
    finally:
      print(''.join(['─']*(len(path)+12)))
  t1 = time.time()

  #from matplotlib import pyplot as plt
  #plt.hist(times, int(np.ceil(max(times)/.00025)), (0,max(times)))

  print('Uitvoeringstijd: %i ms' % int((t1-t0)*1000))


graph = False

if graph:
  from pycallgraph import PyCallGraph
  from pycallgraph.output import GraphvizOutput

  output = GraphvizOutput()
  output.output_file = 'call_graph.svg'
  output.output_type = 'svg'

  with PyCallGraph(output=output):
    execute()
else:
  execute()