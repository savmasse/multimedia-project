# -*- coding: utf-8 -*-

################################################################################
"""
Tekortkomingen: vooral vanwege de optimalisatie naar jigsaws
(er zullen er wel meer zijn)

• Grid:shrink() gaat ervan uit dat ELKE zijde van een puzzelstuk een buur
   heeft, en zal dit steeds verbinden. Wanneer dit gebeurt over een rand heen
   - d.w.z. wanneer interne stukken minder goed matchen met elkaar dan de
     rand met overflowende puzzelstukken - zal het algoritme mogelijk correcte
   verbindingen elimineren van de adjacency matrix door optimalisatie rond het
   foutief verbonden puzzelstuk. Bij jigsaws zijn verbindingen overheen de rand
   onmogelijk omdat de randzijden (= allen vlak) uit de adjacency matrix worden
   verwijderd.

   ⇒ Zorg ervoor dat gewichten van foute matches steeds onder die van goede
     matches ligt. Dit is onmogelijk op voorhand te voorspellen, dus het algo
     kan aangepast worden door puzzelstukken meteen voor een volledige rand uit
     te breiden.

• Bij het verbinden van puzzelstukken wordt geen rekening gehouden met het
   eiland van verbonden puzzelstukken. Dit kan opgelost worden door een lijst
   van onafgewerkte eilandjes bij te houden parallel met de vorming van de
   lijst van werkelijk verbonden puzzelstukken

• Er wordt geen rekening gehouden met de grootte van de puzzel (5x5/2x3/...).
"""
################################################################################

from .exception import Unsolvable, Debug

import numpy as np
from numpy import array as arr, nan

################################################################################
# CONSTANTEN                                                                   #
################################################################################

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

################################################################################
"""
Deze klasse voorziet oplossingsalgoritmen om het masker van mogelijke
verbindingen tussen puzzelstukken te minimaliseren tot een lijst van werkelijk
verbonden puzzelstukken. Vervolgens kan deze lijst gebruikt worden om een 2D
grid mee te genereren.

Voor een 5x5 puzzel van 25 stukken met elk vier randen geldt:

Grid().matrix.shape      == [bool] (25,4,25,4)
Grid().connections.shape == [int]  (25,4)
"""
class Grid: #                                                    2D Puzzelgrid
################################################################################

  # Sla de adjacency matrix op en voorzie lege oplossingscontainers die door
  #  de klassemethoden kruisgewijs gelezen en beschreven worden
  def __init__ ( self, matrix ):
    assert(matrix.dtype==np.bool)
    self.matrix = matrix.copy()  # Adjacency matrix
    self.connections = np.empty((matrix.shape[0], 4, 2))
    self.connections.fill(nan)
    self.find_compatible()

#----------------------------------------------------------------------- Logging

  # TODO: onduidelijke informatie

  # Geef informatie over voor de huidige voortgang in het vormen van een grid
  def print_stats ( self, goal=None ):
    combinations = np.prod(self.matrix.shape[:2])
    candidates = self.matrix.sum()
    per_side = candidates/combinations
    print('%i possible matches down to %.2f per side (%.2f%%)'  % (
      combinations, per_side, 100*per_side/combinations
    ), end=' ')

    if goal is not None:
      print('[%i/%i]' % (candidates,goal))
    else:
      print()

#-------------------------------------------------------------- Adjacency matrix

  @property
  def tot_n_candidates ( self ):
    return self.n_compatible.sum()

  @property
  def finished ( self ):
    #self.find_compatible()
    return self.tot_n_candidates == 0

  def find_compatible ( self ):
    self.n_compatible = self.matrix.sum((2,3))
    self.find_perfect()
    return self.n_compatible

  def find_perfect ( self ): # [!] niet perfect, want soms is 0 verbindingen ideaal
    self.perfect = self.n_compatible == 1
    return self.perfect

  def solve_weights ( self, weights, key='best_weight', pre_shrink=True ):
    """
    KEYS:
      best_weight      => max weight
      least_candidates => minste kandidaten -> max weight
      neighbours       => minste neighbours -> max sum weight/#edges  # TODO, WRS ENORM NUTTIG, niet makkelijk
      mergers          => combineren islands                          # enkel in dromenland
    """

    if pre_shrink:
      self.shrink()

    if key == 'best_weight':
      while not self.finished:
        self.connect(*self.qry_best_weight(weights))
        #self.shrink() GAF SLECHTER RESULTAAT?
    elif key == 'least_candidates':
      while not self.finished:
        self.connect(*self.qry_least_candidates(weights))
        self.shrink()
    elif key == 'expand_border_as_group':
      """
      wts = weights
      for a in range(5):
        for i in range(4):
          try:
            b,j = np.unravel_index(np.nanargmax(wts[a,(i+1)%4]), (25,4))
            c,k = np.unravel_index(np.nanargmax(wts[b,(j+1)%4]), (25,4))
            d,l = np.unravel_index(np.nanargmax(wts[c,(k+1)%4]), (25,4))
            e,m = np.unravel_index(np.nanargmax(wts[d,(l+1)%4]), (25,4))
            if e == a and m == i:
              self.connect(a,(i+1)%4, b,j)
              self.connect(b,(j+1)%4, c,k)
              self.connect(c,(k+1)%4, d,l)
              self.connect(d,(l+1)%4, a,i)
          except:
            continue
        break
      """
      raise Debug('TODO: implementeer expand_border_as_group')
    else:
      raise Exception('apply_weights(key=\'%s\') bestaat niet' % key)

  def qry_least_candidates ( self, weights ):
    # Filter op zijden met het minst aantal compatibele zijden
    lc = np.where(self.n_compatible == self.n_compatible[self.n_compatible != 0].min()) # 0 negeren via underflow
    #print('lc', lc)
    wlc = weights[lc]
    #print('wlc', wlc)
    ilc, b,j = np.unravel_index(np.nanargmax(wlc), wlc.shape)
    a, i = lc[0][ilc], lc[1][ilc]
    return a,i, b,j

  def qry_best_weight ( self, weights ):
    wm = np.where(self.matrix)
    wt = weights[self.matrix]
    i = np.nanargmax(wt)
    return wm[0][i], wm[1][i], wm[2][i], wm[3][i]

#------------------------------------------------------------------ Verbindingen

  def shrink ( self ):
    while self.perfect.any():
      for a,i in zip(*np.where(self.perfect)):
        for b,j in zip(*np.where(self.matrix[a,i])): # 1 coord => max 1 iter
          self.connect(a,i,b,j, _update_cache=False)
      self.find_compatible()

  def connect ( self, a, i, b, j, _update_cache=True ):
    # Verbinding opslaan
    self.connections[a,i] = [b,j]
    self.connections[b,j] = [a,i]

    # Alle andere verbindingen tot deze zijden wegschrappen
    self.matrix[a,i] = False
    self.matrix[b,j] = False
    self.matrix[:,:,a,i] = False
    self.matrix[:,:,b,j] = False

    # TODO: compatibiliteit van connected stukken mergen (steunt op islands)

    # Steeds nodig, tenzij meerdere connecties tegelijkertijd worden
    # > ook daarna moet de cache vernieuwd worden
    if _update_cache:
      self.find_compatible()

#--------------------------------------------------------------- Oplossingsgraaf

  def build ( self, a=None, grid=None, connections=None ):
    connections = connections or self.connections.copy()

    # Begin vanuit het punt met het grootste aantal kandidaat-buren
    a = a or np.isnan(self.connections[:,:,0]).sum(axis=1).argmax()

    grid =  grid or arr([[[a,0]]], dtype=float) # float ondersteunt NaN

    return add_to_grid(a, grid, connections) # recursief


################################################################################
def add_to_grid ( a, grid, connections ):
################################################################################

  # Locate piece A
  loc_a = arr(np.where(grid[:,:,0] == a)).squeeze().T   # x,y (or first piece: x) of piece A
  try: # TODO: fix?
    ori_a = int(grid[(*loc_a, 1)])                  # orientation of A in the puzzle coord system
  except:
    # TODO: is waarschijnlijk niet de juiste uitleg
    raise Unsolvable('Same piece is placed twice in the puzzle, correct size = %i/%i' % (np.isnan(grid).sum()-1, connections.shape[0]), data=grid)

  # Place all neighbours
  #print('connections:')
  #print(connections[a])
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
    #print('-'*10)
    #print(grid[:,:,0])

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
