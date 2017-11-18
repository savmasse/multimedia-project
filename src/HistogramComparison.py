from Matcher import Matcher
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PuzzleSolver import PuzzleSolver as ps

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
        print (matched);
        
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

        # Create histograms in grayscale first
        hist1 = cv2.calcHist([firstPiece], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]);
        hist1 = cv2.normalize(hist1, hist1).flatten();
        hist2 = cv2.calcHist([secondPiece], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256]);
        hist2 = cv2.normalize(hist2, hist2).flatten();
        
        cmp = cv2.compareHist(hist1, hist2, self.comparator);
        
        return cmp;
        
        
# Temporary test of the above class #

image = cv2.imread("../Tiles_shuffled/tiles_shuffled_2x2_00.png", cv2.IMREAD_COLOR);

# Keep this here to test this file independently; must be removed later...
def divideImage(image, dim):
    r = int(len(image)/dim[0]);
    c = int(len(image[0])/dim[1]);
    matrix = np.zeros((dim[0], dim[1], r, c, 3));
    
    for i in range (0, dim[0]):
        for j in range (0, dim[1]):
                # matrix.append(image[i*r:(i+1)*r, j*c: (j+1)*c]);
                matrix[i, j] = image[i*r:(i+1)*r, j*c:(j+1)*c];
    return matrix;

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB);
plt.imshow(image);
plt.show();

hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

matrix = divideImage(image, [2, 2]);
# shuffle correctly and look at result
res = np.copy(matrix);
res[0,0] = matrix[1,1];
res[0,1] = matrix[0,1];
res[1,0] = matrix[0,0];
res[1,1] = matrix[1,0];

hc = HistogramComparison(10, 0.8, cv2.HISTCMP_INTERSECT);

# Check the neighbours of all pieces in the 2x2 puzzle
hc.match(res, 0, 0);
hc.match(res, 0, 1);
hc.match(res, 1, 0);
hc.match(res, 1, 1);

    