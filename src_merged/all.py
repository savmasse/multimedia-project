# -*- coding: utf-8 -*-

###############################################################################

import cv2
import glob
from puzzlesolver import Puzzle, Unsolvable
from puzzlesolver.exception import Debug
from snippets import Time, Count

###############################################################################
""" Copy & Paste bord
HISTCMP_CORREL
HISTCMP_INTERSECT
HISTCMP_CHISQR
HISTCMP_CHISQR_ALT
HISTCMP_BHATTACHARYYA
HISTCMP_KL_DIV
"""

graph = False
filt = {
  'shape' : "",
  'size'  : "",
  'order' : "",
  'n'     : ""
}
blocking = {
  'show_success': False,
  'show_failure': False
}
params = {
  #'methods': [], # enkel shrinken
  'methods': [('histcmp', cv2.HISTCMP_CORREL , 'best_weight')],
}

###############################################################################

def execute ( shape, size, order, n, methods, show_failure=True, show_success=True ):
  # Zoek puzzel afbeeldingen
  query = '*%s*_*%s*_*%s*_*%s*' % (shape, order, size, n)
  paths = sorted(glob.glob('../img/**/%s.png' % query, recursive=True))
  if len(paths): print('%s┐' % ''.join(['─']*(len(paths[0])+11)))
  # Hou statistieken bij
  with Time('Uitvoeringstijd') as exec_timer,\
       Count('Opgeloste puzzels', msg='%%s: %%i/%i'%len(paths)) as n_solved:

    # Overloop de gekozen puzzels
    for path in paths:
      try:
        try:
          msg = '   '

          puzzle = Puzzle(path)

          with Time('Piece extraction'):
            puzzle.extract_pieces()

          with Time('Solution'):
            if show_failure:
              grid = puzzle.solve(radius=2, methods=methods, show_failure=exec_timer)
            else:
              grid = puzzle.solve(radius=2, methods=methods)
            if grid is not False:
              n_solved.count()

          with exec_timer.pause():
            if show_success:
              puzzle.show_color(block=False)
              puzzle.show_solution(grid)
              #puzzle.show_compatibility(puzzle.compatibility_matrix(), weights=puzzle.histogram_correlation_matrix())
        except Exception as exception:
          raise exception
          pass

      except Unsolvable as exc:
        msg = 'EXC'
        print(exc)
      except Debug as exc:
        msg = 'DBG'
        return

      finally:
        print(puzzle.correct(path), msg, path)
        print('%s┤' % ''.join(['─']*(len(path)+11)))

  #from matplotlib import pyplot as plt
  #plt.hist(times, int(np.ceil(max(times)/.00025)), (0,max(times)))
  cv2.destroyAllWindows()
  if len(paths): print('%s┘' % ''.join(['─']*(len(paths[-1])+11)))
if graph:
  from pycallgraph import PyCallGraph
  from pycallgraph.output import GraphvizOutput

  output = GraphvizOutput()
  output.output_file = 'call_graph.svg'
  output.output_type = 'svg'

  with PyCallGraph(output=output):
    execute(**params, **blocking, **filt)
else:
  execute(**params, **blocking, **filt)