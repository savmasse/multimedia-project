# -*- coding: utf-8 -*-

from .exception import Unsolvable, Debug

import numpy as np
from numpy import array as arr, nan

DIR = [
  [ 0, 1],
  [-1, 0],
  [ 0,-1],
  [ 1, 0]
]

PAD = [
  ((0,0), (0,1), (0,0)),
  ((1,0), (0,0), (0,0)),
  ((0,0), (1,0), (0,0)),
  ((0,1), (0,0), (0,0))
]

class Grid:

  def __init__ ( self, matrix ):
    assert(matrix.dtype==np.bool)
    self.matrix = matrix  # bevat geen NaN's
    self.grid = None      # bevat wel NaN's
    self.find_compatible()
    self.connections = np.empty((matrix.shape[0], 4, 2))
    self.connections.fill(nan)
    self.find_compatible()

###############################################################################
#                                  MATRIX                                     #
###############################################################################

  @property
  def finished ( self ):
    # TODO: param recalculate?
    self.find_compatible() # WRS NUTTELOZE OVERHEAD
    return self.n_compatible.sum() == 0

  def find_compatible ( self ):
    self.n_compatible = self.matrix.sum((2,3))
    self.find_perfect()
    return self.n_compatible

  def find_perfect ( self ):
    self.perfect = self.n_compatible == 1
    return self.perfect

  def test ( self ):
    return # TODO
    raise Unsolvable

###############################################################################
#                                  CONNECT                                    #
###############################################################################

  def shrink ( self ):
    while self.perfect.any():
      for a,i in zip(*np.where(self.perfect)):
        for b,j in zip(*np.where(self.matrix[a,i])): # 1 coord => max 1 iter
          #print('%i,%i %i,%i' % (a,i,b,j))
          self.connect(a,i,b,j, _update_cache=False)
      self.find_perfect()

  def connect ( self, a, i, b, j, _update_cache=True ):
    self.connections[a,i] = [b,j]
    self.connections[b,j] = [a,i]

    # Remove connection and competitors for this target
    # > remove competitors
    for c,k in zip(*np.where(self.matrix[a,i])):
      self.matrix[c,k,a,i] = False
      self.n_compatible[c,k] -= 1
    for c,k in zip(*np.where(self.matrix[b,j])):
      self.matrix[c,k,b,j] = False
      self.n_compatible[c,k] -= 1
    # > remove base
    self.matrix[a,i] = False
    self.n_compatible[a,i] = 0
    # > remove mirror
    self.matrix[b,j] = False
    self.n_compatible[b,j] = 0

    if _update_cache:
      self.find_perfect()

  def apply_weights ( self, weights ):
    print('TODO: werken met gewichten')

###############################################################################
#                                   BUILD                                     #
###############################################################################

  def build ( self, a=None, grid=None, connections=None ):
    connections = connections or self.connections.copy()
    a = a or np.isnan(self.connections[:,:,0]).sum(axis=1).argmin() # best verbonden punt
    grid =  grid or arr([[[a,0]]], dtype=float) # float ondersteunt NaN

    print(connections)
    print('building...')
    grid = add_to_grid(a, grid, connections)
    print('built')

    return grid.astype(np.int)
    # TODO: bouwen tot er geen island overblijft

def add_to_grid ( a, grid, connections ): # RECURSIEF
  # TODO: is tip_i wel op side_i?

  # Locate piece A
  loc_a = arr(np.where(grid[:,:,0] == a)).squeeze().T   # x,y (or first piece: x) of piece A
  try: # TODO: fix?
    ori_a = int(grid[(*loc_a, 1)])                  # orientation of A in the puzzle coord system
  except:
    raise Unsolvable('Same piece is placed twice in the puzzle')

  # Place all neighbours
  print('connections:')
  print(connections[a])
  for i, (b,j) in enumerate(connections[a]):
    if not np.isnan([b,j]).all():
      b,j = int(b),int(j)
    else:
      continue

    # Determine orientation and position of the piece to be placed <B>
    # > Orientation
    to_b  = (i-ori_a)%4  # direction A->B in the puzzle coordinate system
    to_a  = (to_b+2)%4   # the reversed to_b direction = rotate 180°
    ori_b = (j-to_a)%4   # orientation of B in the puzzle coord system
    # > Position
    loc_b = loc_a + DIR[to_b]

    # Place the piece in the grid matrix
    # > Extend the matrix if needed
    if (loc_b < 0).any() or (loc_b >= arr(grid.shape[:2])).any():
      grid = np.pad(grid, PAD[to_b], 'constant', constant_values=nan)
      # Coordinaten aanpassen voor nieuwe dimensies
      offset = np.clip(DIR[to_b], 0, 1)
      loc_b = loc_a+offset
    # > Place the piece
    grid[(*loc_b,)] = [b,ori_b]

    # TODO:remove placed from connections (! mirrorred)
    print('-'*10)
    print(grid[:,:,0])

  # TODO: kan dit bij in de vorige loop?
  for i, (b,j) in enumerate(connections[a]):
    if not np.isnan([b,j]).all():
      b,j = int(b),int(j)
    else:
      continue

    # Remove from connections matrix
    connections[a,i] = nan
    connections[b,j] = nan

    grid = add_to_grid( b, grid, connections )

  return grid