from PuzzleSolver import PuzzleSolver
import numpy as np;

class Matcher (object):
    
    def __init__(self):
        pass;
        
    def match(self):
        pass;
        
    def hasNeighbours(self, pieces,row, col):
        """
        Return booleans for existance of the neighbours of a puzzle piece.
        """
        # create list of neighbours: set them all to not exist
        neighbours = np.zeros((4));
        r = len(pieces);
        c = len(pieces);
        
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
        
    