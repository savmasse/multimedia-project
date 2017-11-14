#
# Handler for the solving of regular puzzles. Is also parent class for the 
# solvers of more complex puzzles.
#

import cv2;
import numpy as np;
import matplotlib.pyplot as plt;

class PuzzleSolver(object): 
    # Static variables
    DIR_LEFT = 0;
    DIR_RIGHT = 1;
    DIR_UP = 2;
    DIR_DOWN = 3;
    
    def __init__(self, puzzleImage, puzzlePieces, classifier, shuffler):
        self.puzzleImage = puzzleImage;
        
        if (puzzlePieces is None):
            self.puzzlePieces = self.calculatePieces();
        else:
            self.puzzlePieces = puzzlePieces;
            
        # Set classifier for matching pieces
        self.classifier = classifier;
        # Set shuffler
        self.shuffler = shuffler;
            
            
    def calculatePieces(self):
        """
        If doens't already know the pieces, calculate them.
        """
        pass;
        
    def showOriginalCV(self):
        cv2.imshow("Original puzzle", self.puzzleImage);
        cv2.waitKey();
        cv2.destroyAllWindows();
        
    def showOriginal(self):
        plt.imshow(self.puzzleImage);
        plt.show();
        
    def showAssembled(self):
        """
        Show the assembled image from all pieces in their current order.
        """
        im = np.zeros(self.puzzleImage.shape);
        r,c,d = self.puzzleImage.shape;
        r = r/len(self.puzzlePieces); # assume square matrix
        c = c/len(self.puzzlePieces);
    
        for i in range (len(self.puzzlePieces)):
            for j in range (len(self.puzzlePieces)):
                im[i*r:(i+1)*r, j*c:(j+1)*c] = self.puzzlePieces[i,j];
    
        plt.imshow(im);
        plt.show();
    
    def hasNeighbours(self, row, col):
        """
        Return booleans for existance of the neighbours of a puzzle piece.
        """
        # create list of neighbours: set them all to not exist
        neighbours = np.zeros((4));
        r = len(self.puzzlePieces);
        c = len(self.puzzlePieces[0]);
        
        # now add the ones that do exist
        if (col>0):
            neighbours[PuzzleSolver.DIR_LEFT] = 1;
        if (col<(c-1)):
            neighbours[PuzzleSolver.DIR_RIGHT] = 1;
        if (row>0):
            neighbours[PuzzleSolver.DIR_UP]= 1;
        if (row<(r-1)):
            neighbours[PuzzleSolver.DIR_DOWN] = 1;
        
        # convert to integer and return
        return np.int8(neighbours);
  
    def shuffle(self):
        self.puzzlePieces = self.shuffler.shuffle(self.puzzlePieces);
        
    def match(self):
        return self.classifier.match();
        
        