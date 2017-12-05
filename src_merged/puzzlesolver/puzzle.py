# -*- coding: utf-8 -*-

# TODO: niet enkel contour nemen (sommige afbeeldingen hebben floating islands)

from .exception import Unsolvable, Debug
from .piece import Piece
from .grid import Grid

from snippets import Grid_solution_stats

import cv2
import numpy as np
from numpy import array as arr
from scipy.signal import argrelmax as arm

###############################################################################

blue  = (255,0,0)
green = (0,255,0)
red   = (0,0,255)
redgreen = lambda w: tuple(np.clip([0,512*w,512*(1-w)], 0, 255))

###############################################################################

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

    self.check_size()

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
      rectangles.append(arr(polygon[idx]))

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
      cv2.imshow('segment_tiles(): geen matches', self.input)
      cv2.waitKey(10)
      from matplotlib import pyplot as plt
      plt.figure(figsize=(20,10))
      plt.subplot(2,1,1)
      plt.plot(dx[1:-1])
      plt.scatter(arm_x-2, dx[arm_x], color='red')
      plt.subplot(2,1,2)
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
      Piece().from_slice(
        self.input[ 1+int(y*h) : 1+int((y+1)*h),
                    1+int(x*w) : 1+int((x+1)*w) ],
        offset=[1+int(x*w), 1+int(y*h)]
      )
      for y in range(self.shape[0])
      for x in range(self.shape[1])
    ]

  def check_scrambled ( self ):
    """ Controleer of de puzzelstukken rechtop staan [incl. rotaties n*90°] """

    self.scrambled = False
    # Zoek een schuine lijn tussen de reeds gevonden hoeken [self.corners]
    for corners in self.corners:
      if np.all(np.diff(corners.squeeze(), axis=0), axis=1).sum() > 1:
        self.scrambled = True
        break

  def check_size ( self ):
    vecs = np.diff(np.pad([piece.corners for piece in self.pieces], ((0,0),(0,1),(0,0),(0,0)), 'wrap'), axis=1).squeeze()
    lens = np.hypot(vecs[:,:,0], vecs[:,:,1])

    srtd = np.sort(lens.ravel())
    half = int(len(srtd)/2)

    if abs(srtd[:half][-1]-srtd[half:][0]) < 10:
      sides = arr([srtd[half]]*4) # 0 = same len
      for piece in self.pieces:
        piece.sides = sides
    else:
      short = int(np.rint(np.median(srtd[half:], overwrite_input=True)))
      long  = int(np.rint(np.median(srtd[:half], overwrite_input=True)))
      self.size = (short,long)

      for a in range(len(self.pieces)):
        if abs(lens[a,0]-short) < 10:
          self.pieces[a].sides = arr([long,short,long,short]) # 1 = short
        else:
          self.pieces[a].sides = arr([short,long,short,long]) # 2 = long


  def compatibility_matrix ( self ):
    # Genereer volledig toegelaten output matrix
    n = len(self.pieces)
    matrix = np.ones((n,4, n,4), dtype=bool)

    # Diagonaal niet toelaten
    #  = een puzzelstuk kan zijn eigen randen niet met elkaar matchen
    for ab in range(n):
      matrix[ab,:,ab,:] = False

    # QuickVar
    diag = np.zeros((4,4), dtype=bool)
    np.fill_diagonal(diag, True)

    for a, pa in enumerate(self.pieces):
      for b, pb in enumerate(self.pieces):
        # TODO: diagonaal vullen en mirrorren

        mat = matrix[a,:,b,:]
        # Zijden moeten even lang zijn
        # (+- even lange zijden hebben dezelfde waarde gekregen)
        sidelen_ok = pa.sides[:,None] == pb.sides[None,:]
        np.logical_and(sidelen_ok, mat, out=mat)

        # Jigsaw
        if pa.tips and pb.tips:
          # TODO: zo veel mogelijk logica mergen

          # Beide zijden moeten een Tip hebben
          # > verwijdert platte zijden en incompatibele Tips
          # [+ versnelt uitvoering]
          pat = arr([(1 if tip[0] else -1) if tip is not None else 0.5 for tip in pa.tips])[:,None]
          pbt = arr([(1 if tip[0] else -1) if tip is not None else 0.5 for tip in pb.tips])[None,:]
          tipcompat_ok = np.logical_not(pat+pbt)
          np.logical_and(tipcompat_ok, mat, out=mat)

          # Jigsaw Tips moeten op (ongeveer) dezelfde positie zitten
          # [+ versnelt uitvoering]
          pap = arr([tip[5] if tip is not None else 0 for tip in pa.tips])[:,None]
          pbp = arr([tip[5] if tip is not None else 0 for tip in pb.tips])[None,:]
          # MAX AFWIJKING RATIO V ZIJDE (opgelet, korte zijden hebben een kleinere marge dan lange door relatief te werken...)
          tippos = np.abs(pap+pbp-1)
          tippos_ok = tippos < .02 # TODO: afhankelijk maken van de absolute grootte van het puzzelstuk (en nieit meer relatief werken)
          np.logical_and(tippos_ok, mat, out=mat)

          # De vlakke zijden langs de rand moeten verplicht doorlopen
          # > zou ook kunnen via convolutie achtige methode (met condities?)
          # [- vertraagt uitvoering ] 600ms
          waf = np.where([tip is None for tip in pa.tips])[0]
          wnbf = arr(np.where([tip is not None for tip in pb.tips])[0])

          for i in waf:
            mat[i-1][(wnbf+1)%4] = False
            mat[(i+1)%4][(wnbf+3)%4] = False

    return matrix

  def histogram_correlation_matrix ( self, radius=5, hsv=False, channels=range(3), mask=None, method=cv2.HISTCMP_CORREL ):
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
    # TODO: support voor andere cv2.HISTCMP_ algoritmen
    if mask is not None:
      where = np.where(mask)
      for a,i,b,j in zip(*where):
        matrix[a,i,b,j] = matrix[a,i,b,j] = \
          cv2.compareHist(hists[a][i], hists[b][j], method)
    else:
      for a, a_edges in enumerate(hists):
        for b, b_edges in enumerate(hists):
          if a is not b:
            for i, a_edge_hist in enumerate(a_edges):
              for j, b_edge_hist in enumerate(b_edges):
                matrix[a,i,b,j] = matrix[a,i,b,j] = \
                  cv2.compareHist(a_edge_hist, b_edge_hist, method)

    # TODO: output omzetten naar correcte waarden
    if method == cv2.HISTCMP_CORREL:
      return np.abs(matrix)
    elif method == cv2.HISTCMP_CHISQR or method == cv2.HISTCMP_BHATTACHARYYA or method == cv2.HISTCMP_CHISQR_ALT or method == cv2.HISTCMP_KL_DIV:
      return np.nanmax(matrix)-matrix
    elif method == cv2.HISTCMP_INTERSECT:
      return matrix
    else:
      raise Exception('cv2.compareHist(method=%i) wordt niet ondersteund' % method)

  def solve ( self, print_stats=True, radius=5, methods=[('histcmp', cv2.HISTCMP_CORREL, 'best_weight')], show_failure=False ):
    M_compatible = self.compatibility_matrix()
    grid = Grid(M_compatible)

    """ # zet mij terug aan, was tijdens debuggin. (non-breaking)
    with Grid_solution_stats(grid, 'shrink', print_stats):
      grid.shrink()
    """
    for method, param, key in methods:
      if not grid.finished:
        with Grid_solution_stats(grid, '%s(%s)' % (method,param), print_stats):
          if method == 'histcmp':
            M_histograms = self.histogram_correlation_matrix(mask=M_compatible, radius=radius, method=param) # TODO: gebruik M_compatible masker
            grid.apply_weights(M_histograms, key=key)
          grid.shrink()

    if print_stats:
      grid.print_stats(goal=np.prod(self.shape)*4-np.sum(self.shape)*2)

    if grid.finished:
      try:
        solution = grid.build()
      except Unsolvable as dump:
        if show_failure:
          with show_failure.pause():
            print(dump.data[:,:,0])
            try:
              self.show_compatibility(grid.matrix, M_histograms, block=False)
            except:
              self.show_compatibility(grid.matrix, block=False)
            self.show_solution(dump.data)
        raise dump
      return solution
    else:
      return False

  def show_compatibility ( self, matrix, weights=None, abstract=True, title='Compatibility', block=True ):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.createTrackbar('a * i', title, 0, len(self.pieces)*4-1, self._cb_update_compat_trackbars)
    cv2.createTrackbar('[a] piece', title, 0, len(self.pieces)-1, self._cb_show_compatibility)
    cv2.createTrackbar('[i] side', title, 0, 3, self._cb_show_compatibility)
    self._cb_compat_mat = matrix
    self._cb_compat_win = title
    self._cb_compat_wts = weights
    self._cb_compat_pause = False

    if abstract == True:
      self._cb_compat_src = self.show_info(show=False,block=False)
    else:
      self._cb_compat_src = self.input
    self._cb_show_compatibility(None)

    if block:
      cv2.waitKey()

  def _cb_update_compat_trackbars ( self, ai ):
    win = self._cb_compat_win
    self._cb_compat_pause = True
    cv2.setTrackbarPos('[a] piece', win, ai//4)
    self._cb_compat_pause = False
    cv2.setTrackbarPos('[i] side', win, ai%4)

  def _cb_show_compatibility ( self, _ ):
    if self._cb_compat_pause: return

    src = self._cb_compat_src
    mat = self._cb_compat_mat
    win = self._cb_compat_win
    wts = self._cb_compat_wts

    a = cv2.getTrackbarPos('[a] piece', win)
    i = cv2.getTrackbarPos('[i] side', win)
    cv2.setTrackbarPos('a * i', win, 4*a+i)
    pa = self.pieces[a]
    l = mat[a,i]

    img = src.copy()
    pa.draw_side(img, i, blue, offset=pa.offset, thickness=8)
    for b,j in zip(*np.where(l)):
      pb = self.pieces[b]
      if wts is None:
        pb.draw_side(img, j, green, offset=pb.offset, thickness=4)
      else:
        w = abs(wts[a,i,b,j]) # TODO: altijd abs teruggeven
        pb.draw_side(img, j, redgreen(w), offset=pb.offset, thickness=3+int(4*w))
    cv2.imshow(win, img)

  def show_info ( self, block=True, show=True ):
    img = np.repeat(32*self.mask.astype(np.uint8)[:,:,None], 3, 2)
    for i,p in enumerate(self.pieces):
      p.show_info(img, block=block, view=False, offset=p.offset, stacked=False, id=i)
    if show:
      cv2.namedWindow('Puzzle Info', cv2.WINDOW_NORMAL)
      cv2.imshow('Puzzle Info', img)
    if block:
      cv2.waitKey()
    return img

  def show_solution ( self, grid, m=1, block=True ):
    # Determine piece size
    w,h = self.pieces[int(grid[:,:,0][np.where(grid[:,:,1] == 0)][0])].sides[:2]
    img = np.zeros((int(h*grid.shape[0]+m*(grid.shape[0]+1)), int(w*grid.shape[1]+m*(grid.shape[1]+1)), 3), dtype=np.uint8)

    # Draw pieces
    for y,x in zip(*np.where(~np.isnan(grid[:,:,0]))):
        #for y in range(grid.shape[0]):
        #for x in range(grid.shape[1]):
        p = self.pieces[int(grid[y,x,0])]

        # Punten bepalen voor de rotatie
        pts_from = np.roll(p.corners, int(-grid[y,x,1]), axis=0)[:3,0]#.astype(np.float32)
        pts_to_org = [m+y*(h+m), img.shape[1]-(m+(x+1)*(w+m))] # HACK
        pts_to_dst = [[h,0],[0,0],[0,w]]
        pts_to = (arr(pts_to_org) + arr(pts_to_dst))

        # Transformatie matrix bepalen
        M_transform = cv2.getAffineTransform(np.float32(pts_from), np.float32(pts_to))
        img_piece = cv2.warpAffine(p.input, M_transform, img.shape[:2])

        # HACK ~ zal altijd werken
        if np.equal(img.shape, img_piece.shape).all():
          img += img_piece*(img_piece>0)
        else:
          img_piece = np.rot90(img_piece)
          img += img_piece*(img_piece>0)

    # Weergeven
    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.imshow('output', cv2.convertScaleAbs(img))
    if block:
      cv2.waitKey(0)

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