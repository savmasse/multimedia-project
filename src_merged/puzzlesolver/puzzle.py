# -*- coding: utf-8 -*-

# TODO: niet enkel contour nemen (sommige afbeeldingen hebben floating islands)

from .exception import Unsolvable
from .piece import Piece

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

    # Geen ruimte tussen stukken: automatisch knippen met aangepaste methode
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
        # enkel mogelijk door voorkennis van de set
        # > ALTERNATIEF: - grootste gemene delers die meest naar vierkant neigen?
        #                - puzzel gewoon laten genereren?

      # Tegel of jigsaw herkennen
      # Parameter<2> approxPolyDP: % marge t.o.v. contour lengte om uitstekers te herkennen
      #   3-  = tegels worden fout herkend (meer dan 4 hoeken)
      #   4,5 = tegels = altijd 4 hoeken; jigsaw in 1 pic 4 hoeken
      #   5   = fouten in extract_jigsaws
      #   6+  = jigsaw heeft af en toe meer dan 4 hoeken
      polygons = [cv2.approxPolyDP(contour, 4, True) for contour in self.contours]
      if not (self.extract_tiles(polygons) or self.extract_jigsaws(polygons)):
        raise Unsolvable('Failed to extract pieces')
      else:
        self.check_scrambled()

    # Lengtes bepalen
    vecs = np.diff(np.pad([piece.corners for piece in self.pieces], ((0,0),(0,1),(0,0),(0,0)), 'wrap'), axis=1).squeeze()
    lens = np.hypot(vecs[:,:,0], vecs[:,:,1])

    srtd = np.sort(lens.ravel())
    half = int(len(srtd)/2)

    if abs(srtd[:half][-1]-srtd[half:][0]) < 10:
      sides = [0]*4 # 0 = same len
      for piece in self.pieces:
        piece.sides = sides
    else:
      short = int(np.rint(np.median(srtd[half:], overwrite_input=True)))
      for a in range(len(self.pieces)):
        if abs(lens[a,0]-short) < 10:
          self.pieces[a].sides = [2,1,2,1] # 1 = short
        else:
          self.pieces[a].sides = [1,2,1,2] # 2 = long

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

      # L1,L2 = OVERFIT (.04 ideaal, 0395 1 err, 0378 0 err)
      # SP = SAFE    (max uitchieter = .1308, meest <.1)
      # Vier scherpste hoeken bijhouden (via sortering)
      possibles = sorted(((i,sp)
        for i, (l1,l2,sp,pt) in enumerate(zip(rel_lens[:-1], rel_lens[1:], scal_prods, polygon))
        if l1>.0378 and l2>.0378 and sp < .2
      ), key=lambda x:x[1])

      # SELFTEST
      if len(possibles) < 4:
        return False # gebeurt niet tenzij wijzigingen aan parameters

      # HINT: kan ook via np.argsort()

      # Hersorteren volgens tegenwijzerszin
      idx = sorted([i for i,sp in possibles[:4]])

      # Bijhouden om op te slaan als hoekpunten in de klasse
      rectangles.append(np.array(polygon[idx]))

    # Puzzeltype opslaan
    self.corners = rectangles
    self.tile = False

    # Puzzelstukken genereren
    self.pieces = [
      Piece().from_contour(self.input, self.gray, contour, corners, jigsaw=True)
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
    """ Controleer of de puzzelstukken rechtop staan [incl. rotaties n*90°] """

    self.scrambled = False

    # Zoek een schuine lijn tussen de reeds gevonden hoeken [self.corners]
    for corners in self.corners:
      if np.any(np.all(np.diff(corners.squeeze(), axis=0), axis=1)):
        self.scrambled = True
        break

  def compatibility_matrix ( self ):
    # Genereer volledig toegelaten output matrix
    n_pieces = len(self.pieces)
    matrix = np.ones((n_pieces,4, n_pieces,4), dtype=bool)

    # Diagonaal niet toelaten
    #  = een puzzelstuk kan zijn eigen randen niet met elkaar matchen
    for ab in range(n_pieces):
      matrix[ab,:,ab,:] = False

    return matrix

  def histogram_correlation_matrix ( self, radius=5, hsv=False, channels=range(3), where=False ):
    """ Genereer een matrix die de correlatie tussen histogrammen van elk paar randen berekent """

    # Histogrammen berekenen
    # hists[piece,edge] = histogram
    if not hsv:
      hists = [
        [
          cv2.normalize(cv2.calcHist([piece.input], range(3), piece.edge_neighbours_mask(side, radius), 3*[8], 3*[0,256]), None)
          for side in range(len(piece.corners))
        ]
        for piece in self.pieces
      ]
    else:
      hists = [
        [
          cv2.normalize(cv2.calcHist([cv2.cvtColor(piece.input, cv2.BGR2HSV)], range(2), piece.edge_neighbours_mask(side, radius), 2*[8], [0,180, 0,256]), None)
          for side in range(len(piece.corners))
        ]
        for piece in self.pieces
      ]

    # Genereer met NaN gevulde output matrix
    n_pieces = len(self.pieces)
    matrix = np.empty((n_pieces,4, n_pieces,4))
    matrix.fill(np.nan) # controleer of enkel de diagonaal achteraf NaN blijft

    # Vul matrix met correlatie van histogrammen
    # TODO: support voor andere cv2.HISTCMP_ algoritmen [! output betekenis wijzigt !]
    if where: # TODO: dit is trager, terwijl het sneller zou moeten zijn (cache misses/where generatie/steeds herindexeren?)
      for a,i,b,j in zip(*where):
        matrix[a,i,b,j] = matrix[a,i,b,j] = \
          cv2.compareHist(hists[a][i], hists[b][j], cv2.HISTCMP_CORREL)
    else:
      for a, a_edges in enumerate(hists):
        for b, b_edges in enumerate(hists):
          if a is not b:
            for i, a_edge_hist in enumerate(a_edges):
              for j, b_edge_hist in enumerate(b_edges):
                matrix[a,i,b,j] = matrix[a,i,b,j] = \
                  cv2.compareHist(a_edge_hist, b_edge_hist, cv2.HISTCMP_CORREL)
    return matrix

  def solve ( self ):
    M_compatible = self.compatibility_matrix()
    M_histograms = self.histogram_correlation_matrix(where=np.where(M_compatible)) # TODO: gebruik M_compatible masker

    n_pieces = len(self.pieces)
    n_top = 10

    for a in range(n_pieces):
      for i in range(4):
        cmp = M_histograms[a,i]
        asrt = np.argsort(cmp, None)[:-4] # 4x NaN
        i_top = np.unravel_index(asrt[-n_top:], cmp.shape)

        #if top[-2]/top[-1] < .9:
        b,j = i_top[0][-1], i_top[1][-1]
        """
        pa = self.pieces[a]
        ema = pa.edge_neighbours_mask(i,5)
        ima = pa.input.copy()
        ima[ema.astype(bool)] = [0,255,0]
        pb = self.pieces[b]
        emb = pb.edge_neighbours_mask(j,5)
        imb = pb.input.copy()
        imb[emb.astype(bool)] = [0,255,0]
        cv2.imshow('piece a', ima)
        cv2.imshow('piece b', imb)
        cv2.waitKey(0)
        """

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
