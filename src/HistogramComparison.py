from Matcher import Matcher
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PuzzleSolver import PuzzleSolver as ps

"""
Classifier for histogram comparison. You can set the comparator you want to use
in the constructor. Best use Hellinger because it has greater distance between 
good and bad matches => easier to set a threshold.
"""

class HistogramComparison (Matcher):
    
    def __init__(self, matchSize, threshold, comparator):
        super(HistogramComparison, self).__init__();
        self.matchSize = matchSize;
        self.threshold = threshold;
        self.comparator = comparator;
        
    def match(self, pieces, row, col):
        neighbours = self.hasNeighbours(pieces, row, col);
        matched = np.zeros((4));
        r, c, d = pieces[0,0].shape;
        
        if (neighbours[ps.DIR_LEFT]):
            matched[ps.DIR_LEFT] = self.checkHistogram(pieces[row, col][:, 0:self.matchSize], pieces[row, col-1][:, (c-self.matchSize):]);
            
        if (neighbours[ps.DIR_RIGHT]):
            matched[ps.DIR_RIGHT] = self.checkHistogram(pieces[row, col][:, (c-self.matchSize):], pieces[row, col+1][:, 0:self.matchSize]);
            
        if (neighbours[ps.DIR_UP]):
            matched[ps.DIR_UP] = self.checkHistogram(pieces[row, col][0: self.matchSize, :], pieces[row-1, col][(r-self.matchSize):, :]);
        
        if (neighbours[ps.DIR_DOWN]):
            matched[ps.DIR_DOWN] = self.checkHistogram(pieces[row, col][(r-self.matchSize):, :], pieces[row+1, col][0:self.matchSize, :]);
        
        # Assume that a perfect match is not possible because compared pieces cannot be identical
        return matched;
        
    def checkHistogram(self, firstPiece, secondPiece):
        # convert to int first
        firstPiece = np.uint8(firstPiece);
        secondPiece = np.uint8(secondPiece);
        
#        # show pieces
#        plt.imshow(firstPiece);
#        plt.title("First piece:");
#        plt.show();
#        plt.imshow(secondPiece);
#        plt.title("Second piece:");
#        plt.show();

        hist1 = cv2.calcHist([firstPiece], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]);
        hist1 = cv2.normalize(hist1, hist1).flatten();
        hist2 = cv2.calcHist([secondPiece], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]);
        hist2 = cv2.normalize(hist2, hist2).flatten();
        
        cmp = cv2.compareHist(hist1, hist2, self.comparator);
        
        return cmp;
    
    def matchNeighbour(self, pieceOne, pieceTwo, direction):
        r, c, d = pieceOne.shape;
        
        if (direction == ps.DIR_LEFT):
            return self.checkHistogram(pieceOne[:, 0:self.matchSize], pieceTwo[:, (c-self.matchSize):]);
            
        if (direction == ps.DIR_RIGHT):
            return self.checkHistogram(pieceOne[:, (c-self.matchSize):], pieceTwo[:, 0:self.matchSize]);
            
        if (direction == ps.DIR_UP):
            return self.checkHistogram(pieceOne[0: self.matchSize, :], pieceTwo[(r-self.matchSize):, :]);
        
        if (direction == ps.DIR_DOWN):
            return self.checkHistogram(pieceOne[(r-self.matchSize):, :], pieceTwo[0:self.matchSize, :]);
