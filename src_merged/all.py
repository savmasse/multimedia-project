# -*- coding: utf-8 -*-

import glob
import time
from puzzlesolver import Puzzle, Unsolvable
from puzzlesolver.exception import Debug
import numpy as np

print('>─────────────────────────────────────────────────────────────<')

def execute ():
  # Kies welke puzzels je wil laden
  paths = []
  #paths.extend(glob.glob('../img/**/*tile*.png', recursive=True))
  #paths.extend(glob.glob('../img/**/*jig*scr*3x3*.png', recursive=True))
  #paths.extend(glob.glob('../img/**/*jigsaw*scrambled*5x5*.png', recursive=True))
  paths.extend(glob.glob('../img/**/*jigsaw*2x3*.png', recursive=True))
  #paths.extend(glob.glob('../img/**/*.png', recursive=True))

  times = []
  n_solved = 0
  t0 = time.time()
  for path in sorted(paths):
    puzzle = Puzzle(path)
    try:
      ta = time.time()
      puzzle.extract_pieces()
      dt = time.time()-ta
      times.append(dt)
      grid_solution = puzzle.solve()
      if grid_solution is not False:
        n_solved += 1
        puzzle.show_solution(grid_solution)

      print(puzzle.correct(path), '%.02f' % np.round(1000*(dt), 2), path)
      #puzzle.show_info(False)
      #puzzle.show_compatibility(solution, weights=puzzle.histogram_correlation_matrix())
    except Unsolvable as exc:
      print(puzzle.correct(path), 'ERR ', path)
      print(exc)
      continue
    except Debug as exc:
      return
    finally:
      print(''.join(['─']*(len(path)+12)))
  t1 = time.time()

  #from matplotlib import pyplot as plt
  #plt.hist(times, int(np.ceil(max(times)/.00025)), (0,max(times)))

  print('Uitvoeringstijd: %i ms' % int((t1-t0)*1000))
  print('Successvol opgelost: %i (%.2f%%)' % (n_solved, 100*n_solved/len(paths)))

  import cv2
  cv2.destroyAllWindows()


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