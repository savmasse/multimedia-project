# -*- coding: utf-8 -*-

################################################################################

import cv2
from puzzlesolver import Picker, Puzzle
from puzzlesolver.exception import Unsolvable, Debug
from snippets import Time, Count

################################################################################
# COPY & PASTE                                                                 #
################################################################################
"""
HISTCMP_CORREL
HISTCMP_INTERSECT
HISTCMP_CHISQR
HISTCMP_CHISQR_ALT
HISTCMP_BHATTACHARYYA
HISTCMP_KL_DIV
"""
################################################################################
# CONFIGURATIE                                                                 #
################################################################################

# Genereer een soort van flowchart
generate_call_graph = False
call_graph_filename = 'call_graph.svg'

# Filter op puzzel type
filt = {
  'shape' : "",
  'size'  : "",
  'order' : "",
  'n'     : ""
}

# Resultaatweergave
blocking = {
  'show_success': False,
  'show_failure': False
}

# Oplossingsmethode
params = {
  #'methods': [], # enkel shrinken
  'methods': [('histcmp', cv2. HISTCMP_CORREL, 'best_weight')],
}

################################################################################
def execute ( #                                                   Main functie
  impaths,
  shape, size, order, n, methods,
  show_failure=True, show_success=True
):
################################################################################

  # Mooie console output
  if len(impaths): print('%s┐' % ''.join(['─']*(len(impaths[0])+11)))

  # Hou statistieken bij
  with Time('Uitvoeringstijd') as exec_timer,\
       Count('Opgeloste puzzels', msg='%%s: %%i/%i'%len(impaths)) as n_solved:

    # Overloop de gekozen puzzels
    for path in impaths:

      # Vang onoplosbare puzzels op
      try:
        msg = '   '

        # Laad de puzzel in het geheugen
        puzzle = Puzzle(path)

        # Detecteer de puzzelstukken uit de originele afbeelding
        with Time('Piece extraction'):
          puzzle.extract_pieces()

        # Los de puzzel op
        with Time('Solution'):
          if show_failure:
            grid = puzzle.solve(radius=2, methods=methods, show_failure=exec_timer)
          else:
            grid = puzzle.solve(radius=2, methods=methods)
          if grid is not False:
            n_solved.count()

        # Geef het resultaat weer
        with exec_timer.pause():
          if show_success:
            puzzle.show_color(block=False)
            puzzle.show_solution(grid)

      # Puzzel is niet oplosbaar volgens het op dat moment lopende algoritme
      except Unsolvable as exc:
        msg = 'EXC'
        print(exc)

      # Stop het programma op een Debug punt   (soort van soft-breakpoint)
      except Debug as exc:
        msg = 'DBG'
        return

      # Print de resultaten uit
      finally:
        print(puzzle.correct(path), msg, path)
        print('%s┤' % ''.join(['─']*(len(path)+11)))

  #from matplotlib import pyplot as plt
  #plt.hist(times, int(np.ceil(max(times)/.00025)), (0,max(times)))

  # Script mooi afsluiten
  cv2.destroyAllWindows()
  if len(impaths): print('%s┘' % ''.join(['─']*(len(impaths[-1])+11)))

################################################################################
def run ( impaths ):
################################################################################

  # Script gewoon uitvoeren
  if not generate_call_graph:
    execute(impaths, **params, **blocking, **filt)

  # Script uitvoeren met profiler (-> output file in zelfde directory)
  else:
    # Vermijd 'No module named pycallgraph' bij gewone uitvoering v/h script
    from pycallgraph import PyCallGraph
    from pycallgraph.output import GraphvizOutput

    # Grafiek eigenschappen hier (zie docs voor véél meer opties)
    output = GraphvizOutput()
    output.output_file = call_graph_filename
    output.output_type = 'svg'

    # Script uitvoeren in context van de profiler
    with PyCallGraph(output=output):
      execute(impaths, **params, **blocking, **filt)



################################################################################
if __name__ == '__main__': #                                  Script uitvoeren
################################################################################

  picker = Picker(run, root='../img', types=[
    '2x2', '2x3', '3x3', '4x4', '5x5',
    'jigsaw', 'tiles',
    'shuffled', 'rotated', 'scrambled'
  ])
