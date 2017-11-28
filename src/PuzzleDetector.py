#
# This class is used to detect which sort of puzzle we're dealing with. We try
# to minimize the properties of the image that must be known beforehand. Attempt
# to determnine the dimensions of the puzzle, and whether it is a normal puzzle
# or a jigsaw puzzle. Also must check if connected pieces or if scrambled pieces.
#

import cv2;
import numpy as np;
import matplotlib.pyplot as plt;
from PuzzleSolver import PuzzleSolver as PuzzleSolver;
from JigsawSolver import JigsawSolver as JigsawSolver;
from HistogramComparison import HistogramComparison;

class PuzzleDetector(object):
    # static variables
    REGULAR_PUZZLE = 0;
    JIGSAW_PUZZLE = 1;
    
    def __init__(self, puzzleImage):
        self.puzzleImage = puzzleImage;
        r,c,d = puzzleImage.shape;
        self.puzzleType = -1;
            
    def dividePuzzlePieces(self, rows, cols):
        """ 
        Divide the puzzle into pieces.
        """
        image = self.puzzleImage;
        
        r = int(len(image)/rows);
        c = int(len(image[0])/cols);
        matrix = np.zeros((rows, cols, r, c, 3));
    
        for i in range (0, rows):
            for j in range (0, cols):
                matrix[i, j] = image[i*r:(i+1)*r, j*c:(j+1)*c];
                
        return matrix;
        
    def calculateDimensions(self):
        """ 
        Get puzzle dimensions by examining the image.
        """
        # Init
        dim = -1;
        
        # Get image contours
        puzzle_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY);
        #contours = cv2.findContours((puzzle_gray != 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1];
        contours = cv2.findContours((puzzle_gray != 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[1];

        # Look at contours to determine amount of pieces
        dim = np.sqrt( len(contours) );
        
        # Handle special case if tiled puzzle with connected pieces
        if (dim > 1 and dim%1 == 0):
            # In this case the puzzle type is unknown !
            dim = int(dim);
            print ("Disconnected puzzle of size (" + str(dim) + "," + str(dim) + ")");
        elif (dim%1 != 0):
            print ("Special case : not a square puzzle ! ");
        elif (dim == 1):
            r, c = self.calcConnectedTilePuzzleDimensions();
            print ("Connected tiled puzzle of size (" + str(r) + "," + str(c) + ")");
            
    def calcConnectedTilePuzzleDimensions(self):
        m = HistogramComparison(3, 1, cv2.HISTCMP_CORREL);
        av = np.zeros((4,4))

        for i in range (2, 6):
            for j in range (2, 6):
                pieces = self.dividePuzzlePieces(i, j);
                local_av = np.zeros((i, j));
        
                # Now find the 'worst' match possible, that means we've found the 
                # border between two pieces.
                for r in range (0, i):
                    for c in range (0, j):
                        res = m.match(pieces, r, c);
                        local_av[r,c] = np.max(res);
                        
                av[i-2, j-2] = np.min(local_av);
                
        #print (av)
                
        rows = np.where(av==av.min())[0][0] + 2;
        cols = np.where(av==av.min())[1][0] + 2;

        return (rows, cols);
    
        
#################
# Detector test #
#################

# Load image
image = cv2.imread("../Tiles_shuffled/tiles_shuffled_3x3_00.png", cv2.IMREAD_ANYCOLOR);
#image = cv2.imread("../Jigsaw_shuffled/jigsaw_shuffled_3x3_00.png", cv2.IMREAD_ANYCOLOR);

p = PuzzleDetector(image);

# print the calculated dimensions
plt.imshow(image);
plt.show();

p.calculateDimensions();

print ("  ");
for i in range (0,9):
    image = cv2.imread("../Tiles_shuffled/tiles_shuffled_4x4_0" + str(i) + ".png", cv2.IMREAD_ANYCOLOR);
    print(str(i) + ": ");
    
    p = PuzzleDetector(image);
    p.calculateDimensions();