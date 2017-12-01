import numpy as np;
import matplotlib.pyplot as plt;
import itertools

""" 
Abstract class for all shuffler classes.
"""
class Shuffler (object):
    
    def __init__(self, puzzlePieces):
        self.puzzlePieces = puzzlePieces;
        
        # Get the flattened image
        self.flattened = 0;
        self.flatten();
        
    def showFlattened(self):
        """
        Plot the puzzle in flattened state.
        """
        l = len(self.flattened);
        im = self.flattened[0]
        for i in range (1, l):
            im = np.hstack((im, self.flattened[i]));
            
        plt.imshow(im);
        plt.show();
    
    def flatten(self):
        """
        Flatten puzzle into one-dimensional array of pieces.
        """
        dimx,dimy,r,c,d = self.puzzlePieces.shape;
        self.flattened = self.puzzlePieces.reshape((dimx*dimy, r, c, d));
        
    def unFlatten(self):
        """
        Restore puzzle to original shape.
        """
        dimx, dimy, r, c, d = self.puzzlePieces.shape;
        im = np.zeros(self.puzzlePieces.shape);
        
        for i in range (len(self.flattened)):
            x = i // dimx;
            y = i % dimx;
            im[x,y] = self.flattened[i];
            
        return im;
    
    def getFlattened (self):
        return self.flattened;


"""
Brute force implementation of shuffling the puzzle. This shuffler goes through
all possible permutations one by one without memory of past moves. This shuffler
is not recommended for puzzles larger than 3x3.
"""
class BruteForceShuffler (Shuffler):
    
    def __init__(self, puzzlePieces):
        super(BruteForceShuffler, self).__init__(puzzlePieces);
        self.permutationIndex = 0;
        
        # Get a list of all possible permutations
        indexes = np.arange (len(self.flattened));
        print ("Calculating possible permutations...");
        self.permutations = list(itertools.permutations(indexes, len(indexes)));
        print ("Done calculating permutations.");
        
    def shuffleFlattened(self):
        #np.random.shuffle(self.flattened);
        
        # Attempt to iterate through all the possible permutations of the indexes
        self.flattened = self.flattened [list(self.permutations[self.permutationIndex])];        
        self.permutationIndex = self.permutationIndex + 1;
        
    def getMaxAttempts (self):
        return len (self.permutations);
                