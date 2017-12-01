# -*- coding: utf-8 -*-

# TODO: niet enkel contour nemen (sommige afbeeldingen hebben floating islands)

from .exception import Unsolvable
from .piece import Piece

import time

import cv2
import numpy as np
from scipy.signal import argrelmax as arm

class Puzzle:

  def __init__ ( self, path ):
    """ Laad een afbeelding en genereer grayscale, binair masker en contours """

    # Laad de puzzel in
    raw = cv2.imread(path)

    # Zero padding (1px) voor contour herkenning (<-> pieces: geen zero padding)
    self.input = cv2.copyMakeBorder(raw, 1,1,1,1, cv2.BORDER_CONSTANT, value=(0,0,0))

    # Grayscale, masker, contours
    self.gray = cv2.cvtColor(self.input, cv2.COLOR_BGR2GRAY)
    self.mask = self.gray != 0
    self.contours = [cnt for cnt in cv2.findContours(np.uint8(self.mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1] if cv2.contourArea(cnt) > 100]

    # OPGELET! verwijderde contours bij zwevende jigsaw particles
    # MAAR: niet oplossen door met morphologyex fill, of self.contours produceert
    #       fouten
    # DUS: bounding box van contour meepakken en later opnieuw contouren berekenen?

    self.tile = None      # TRUE/FALSE
    self.shape = None     # (x,y)
    self.scrambled = None # TRUE<scrambled>/FALSE<shuffled/rotated90°>
    self.pieces = None

  def extract_pieces ( self ):
    """ Haal puzzelstukken uit de input afbeelding en bepaal puzzeltype """

    n_pieces = len(self.contours)

    # Geen ruimte tussen stukken: automatisch knippen bij scherpe overgangen
    if n_pieces is 1:
      self.tile = True
      self.scrambled = False
      self.segment_tiles()

    # Ruimte tussen puzzelstukken: gedetecteerde contours zijn voldoende
    else:
      # Puzzelgrid dimensies bepalen
      n_square = np.sqrt(n_pieces)
      if n_square.is_integer():
        self.shape = (n_square, n_square)
      else:
        self.shape = (2,3)

      # Tegel of jigsaw herkennen
      #   3-  = tegels worden fout herkend (meer dan 4 hoeken)
      #   4,5 = tegels = altijd 4 hoeken; jigsaw in 1 pic 4 hoeken
      #   5   = fouten in extract_jigsaws
      #   6+  = jigsaw heeft af en toe 4 hoeken
      polygons = [cv2.approxPolyDP(contour, 4, True) for contour in self.contours]
      if not (self.extract_tiles(polygons) or self.extract_jigsaws(polygons)):
        raise Unsolvable('Failed to extract pieces')
      else:
        self.check_scrambled()

  def extract_tiles ( self, polygons ):
    """ Haal tegels uit de contouren """

    # Controleer of er exact 4 zijden zijn
    if np.all([len(polygon) == 4 for polygon in polygons]):
      # Puzzeltype opslaan
      self.tile = True
      self.corners = polygons

      # Puzzelstukken genereren
      self.pieces = [
        Piece().from_contour(self.input, self.gray, contour, corners)
        for contour, corners in zip(self.contours, self.corners)
      ]
      return True
    else:
      return False

  def extract_jigsaws ( self, polygons ):
    """ Haal jigsaws uit de contouren """

    # De 90° hoeken vinden
    # > hoek bepalen m.b.v. scalair product van eenheidsvectoren langsheen de zijden (sorted)
    # > thresholden voor minimale zijde lengten (afstelling = ZEER precies / ietwat onstabiel)
    rectangles = []
    for polygon in polygons:
      perimeter = cv2.arcLength(polygon, True)
      extended = np.pad(polygon.squeeze(), ((1,1),(0,0)), 'wrap')
      vectors = np.diff(extended, axis=0)
      rel_lens = np.hypot(vectors[:,0], vectors[:,1])/perimeter
      norms = np.linalg.norm(vectors, axis=1)
      unit_vectors = vectors/norms.reshape(-1,1) # unit vectors
      scal_prods = np.abs(np.sum(unit_vectors[:-1]*unit_vectors[1:], axis=1))

      # L1 = OVERFIT (.04 ideaal, 0395 1 err, 0378 0 err)
      # L2 = OVERFIT (.04 ideaal)
      # SP = SAFE    (max uitchieter = .1308, meest <.1)
      possibles = sorted(((i,sp)
        for i, (l1,l2,sp,pt) in enumerate(zip(rel_lens[:-1], rel_lens[1:], scal_prods, polygon))
        if l1>.0378 and l2>.0378 and sp < .2
      ), key=lambda x:x[1])

      # SELFTEST (mag weg op einde v project)
      if len(possibles) < 4:
        return False # gebeurt niet tenzij wijzigingen aan parameters

      # Vier scherpste hoeken bijhouden (via sortering)
      indices = sorted([i for i,sp in possibles[:4]])

      # Hersorteren volgens tegenwijzerszin
      corners = np.array([polygon[i] for i in indices])

      # Bijhouden om op te slaan als hoekpunten in de klasse
      rectangles.append(corners)

    # Puzzeltype opslaan
    self.corners = rectangles
    self.tile = False

    # Puzzelstukken genereren
    self.pieces = [
      Piece().from_contour(self.input, self.gray, contour, corners)
      for contour, corners in zip(self.contours, self.corners)
    ]

    return True

  def segment_tiles ( self ):
    img = self.gray[1:-1,1:-1].astype(np.int16)
    dx = cv2.convertScaleAbs(np.diff(img, axis=1)).sum(axis=0)
    dy = cv2.convertScaleAbs(np.diff(img, axis=0)).sum(axis=1)

    arm_x = arm(dx, order=int(self.input.shape[1]/5)-2, mode='clip')[0]+1 # MAX PUZZEL SHAPE = 5
    arm_y = arm(dy, order=int(self.input.shape[0]/5)-2, mode='clip')[0]+1 # MAX PUZZEL SHAPE = 5

    h,w = img.shape
    possible_widths  = [(n,np.rint(l)) for l,n in [(w/n,n) for n in range(2,6)] if l==np.rint(l)]
    possible_heights = [(n,np.rint(l)) for l,n in [(h/n,n) for n in range(2,6)] if l==np.rint(l)]

    ratio_caught_x = [(n,r) for n,r in [(n,sum([1 for x in arm_x if x%p==0])/(n-1)) for n,p in possible_widths]  if r>0]
    ratio_caught_y = [(n,r) for n,r in [(n,sum([1 for y in arm_y if y%p==0])/(n-1)) for n,p in possible_heights] if r>0]

    try:
      self.shape = (
        max(reversed(ratio_caught_y), key=lambda nr: nr[1])[0],
        max(reversed(ratio_caught_x), key=lambda nr: nr[1])[0]
      )
    except:
      """
      cv2.imshow('segment_tiles(): geen matches', cv2.convertScaleAbs(img))
      cv2.waitKey(10)
      """

      if not len(ratio_caught_x):
        print('Geen verticale segment scheidingslijnen kunnen matchen')
        print('Kandidaten:', ', '.join(str(a) for a in arm_x))
        print('Veelvouden:', ', '.join('{%i → %i}' % np for np in possible_widths) )
      else:
        print('Langs de verticale as:', ', '.join('{%i → %.2f}' % nr for nr in ratio_caught_x))
      if not len(ratio_caught_y):
        print('Geen horizontale segment scheidingslijnen kunnen matchen')
        print('Kandidaten:', ', '.join(str(a) for a in arm_y))
        print('Veelvouden:', ', '.join('{%i → %i}' % np for np in possible_heights) )
      else:
        print('Langs de horizontale as:', ', '.join('{%i → %.2f}' % nr for nr in ratio_caught_y))

      """
      from matplotlib import pyplot as plt
      plt.figure(figsize=(20,10))
      plt.subplot(1,2,1)
      plt.plot(dx[1:-1])
      plt.scatter(arm_x-2, dx[arm_x], color='red')
      plt.subplot(1,2,2)
      plt.plot(dy[1:-1])
      plt.scatter(arm_y-2, dy[arm_y], color='red')
      plt.show(block=True)
      cv2.destroyAllWindows()
      """
      raise Unsolvable('Verdeling in segmenten is mislukt')

      # TODO: code Sam oproepen uit extern bestand

    # Puzzelstukken genereren
    h,w = [img.shape[i]/self.shape[i] for i in range(2)]
    self.pieces = [
      Piece().from_slice(self.input[1+int(y*h):1+int((y+1)*h),1+int(x*w):1+int((x+1)*w)])
      for y in range(self.shape[0])
      for x in range(self.shape[1])
    ]

  def check_scrambled ( self ):
    """ Controleer of de puzzelstukken rechtop staan (of rotaties n*90°) """

    self.scrambled = False

    # Zoek een schuine lijn tussen de reeds gevonden hoeken [self.corners]
    for corners in self.corners:
      if np.any(np.all(np.diff(corners.squeeze(), axis=0), axis=1)):
        self.scrambled = True
        break

  def solve ( self, radius=5 ):
    t0 = time.time()

    # TODO: ondersteun HSV
    pieces_edge_hists = [
      [
        cv2.normalize(cv2.calcHist([piece.input], range(3), piece.edge_neighbours_mask(side, radius), 3*[8], 3*[0,256]), None)
        for side in range(len(piece.corners))
      ]
      for piece in self.pieces
    ]

    n = len(self.pieces)
    comparison = np.empty((n,4,n,4))
    comparison.fill(np.nan)
    for a, a_edges in enumerate(pieces_edge_hists):
      for b, b_edges in enumerate(pieces_edge_hists):
        if a is not b:
          for i, a_edge_hist in enumerate(a_edges):
            for j, b_edge_hist in enumerate(b_edges):
              comparison[a,i,b,j] = comparison[a,i,b,j] = \
                cv2.compareHist(a_edge_hist, b_edge_hist, cv2.HISTCMP_CORREL)

    a,i = 0,0
    candidates = comparison[a,i]
    i_min = np.nanargmin(candidates)



    print('%i/%i' % (i_min,candidates.size))


    #cv2.imshow('piece', self.pieces[a].input*np.repeat(self.pieces[a].edge_neighbours_mask(i, 5)[:,:,None], 3, axis=2))
    #cv2.waitKey()

    t1 = time.time()
    print('Extracted edges in %.2f ms' % (1000*(t1-t0)))

  def show_color ( self, block=True, title='Puzzle [color]' ):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, self.input)
    if block:
      cv2.waitKey()

  def show_mask ( self, block=True, title='Puzzle [mask]' ):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, self.mask)
    if block:
      cv2.waitKey()

  def show_gray ( self, block=True, title='Puzzle [gray]' ):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, self.gray)
    if block:
      cv2.waitKey()

  def correct ( self, path ):
    return ' '.join([
      '◌' if self.tile is None else '●' if (self.tile and 'tile' in path or (not self.tile) and 'jigsaw' in path) else '◯',
      '◌' if self.shape is None else '●' if ('%ix%i' % self.shape) in path else '◯',
      '◌' if self.scrambled is None else '●' if self.scrambled and 'scrambled' in path or (not self.scrambled) and (not 'scrambled' in path) else '◯'
    ])
