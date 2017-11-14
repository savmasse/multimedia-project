#
# This class is used to detect which sort of puzzle we're dealing with. We try
# to minimize the properties of the image that must be known beforehand. Attempt
# to determnine the dimensions of the puzzle, and whether it is a normal puzzle
# or a jigsaw puzzle. Also must check if connected pieces or if scrambled pieces.
#

import cv2;
import numpy as np;
from PuzzleSolver import PuzzleSolver as PuzzleSolver;
from JigsawSolver import JigsawSolver as JigsawSolver;

class PuzzleDetector(object):
    # static variables
    REGULAR_PUZZLE = 0;
    JIGSAW_PUZZLE = 1;
    
    def __init__(self, puzzleImage):
        self.puzzleImage = puzzleImage;
        r,c,d = puzzleImage.shape;
        self.puzzleType = -1;
            
    def detectPieces(self):
        """ 
        Detect and seperate pieces of the puzzle.
        """
        pass;
        
    def calculateDimensions(self):
        """ 
        Get puzzle dimensions based on puzzle type.
        """
        # Init
        rows = -1, cols = -1;
        
        # If jigsaw, use contours to decide on dimensions.
        
        # If regular puzzle, use Hough (?)
        
        return [rows, cols];
    
    def detectPuzzleType(self):
        """
        Detect whether puzzle is a regular puzzle or a jigsaw.
        """
        if (True):
            return PuzzleDetector.JIGSAW_PUZZLE;
        else:
            return PuzzleDetector.REGULAR_PUZZLE;
        
    def printResult(self):
        self.p.printResult();
        
        
#################
# Detector test #
#################

# Load image
image = cv2.imread("../Tiles_shuffled/tiles_shuffled_2x2_00.png", cv2.IMREAD_ANYCOLOR);
p = PuzzleDetector(image);

# print the calculated dimensions
print (p.calculateDimensions());        
        