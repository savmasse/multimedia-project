from HistogramComparison import HistogramComparison
import cv2;
import numpy as np
import matplotlib.pyplot as plt;
from Shuffler import BruteForceShuffler
import timeit;
import math;
import itertools;
from PuzzleSolver import PuzzleSolver as ps;



def showComplete(matrix, image, title = 'Image'):
    im = np.zeros(image.shape);
    r, c, d = image.shape
    r = int (r/len(matrix));
    c = int (c/len(matrix));
    
    for i in range (len(matrix)):
        for j in range (len(matrix[0])):
            im[i*r: (i+1)*r, j*c:(j+1)*c] = matrix [i,j];
            
    plt.figure(figsize=(20,10));
    plt.imshow(im);
    plt.title(title);
    plt.show()
    
    
def divideImage(image, dim):
    r = int(len(image)/dim[0]);
    c = int(len(image[0])/dim[1]);
    matrix = np.zeros((dim[0], dim[1], r, c, 3));
    
    for i in range (0, dim[0]):
        for j in range (0, dim[1]):
                # matrix.append(image[i*r:(i+1)*r, j*c: (j+1)*c]);
                matrix[i, j] = image[i*r:(i+1)*r, j*c:(j+1)*c];
    return matrix;

#
## Try to match a simple 2x2 image
#image = cv2.imread("../Tiles_shuffled/tiles_shuffled_3x3_01.png");
#dim = [3,3];
#
#matrix = divideImage(image, dim);
#hc = HistogramComparison(2, 0.8, cv2.HISTCMP_HELLINGER);

#print ("Match shuffled matrix: ");
#for i in range (0,2):
#    for j in range (0,2):
#        print(hc.match(matrix, i, j));
        
## Correct matrix for image_00
#res = np.copy(matrix);
#res[0,0] = matrix[1,1];
#res[0,1] = matrix[0,1];
#res[1,0] = matrix[0,0];
#res[1,1] = matrix[1,0];

#shuffle = BruteForceShuffler(matrix);
#shuffle.flatten();
#
#for i in range (1):
#    shuffle.shuffleFlattened();
#    pieces = shuffle.unFlatten();
#        
#    av = 0;
#    vals = np.zeros((dim[0],dim[1]));
#    for j in range (dim[0]):
#        for k in range (dim[1]):
#            m = np.mean(hc.match(pieces, j, k), dtype = np.float16);
#            vals[j,k] = m;
#            av = av + m;
#            
#    av = av/(dim[0]*dim[1]);
#    if (av < 0.55):
#        showComplete(pieces, image);
#        print (av);
#        print (vals);


#print ("\nMatch correct image: ");
#for i in range (0,2):
#    for j in range (0,2):
#        print (hc.match(res, i, j));

def match2x2 (image, dim):
    # Divide into pieces
    matrix = divideImage(image, dim);
    
    # Create a new histogram comparator
    hc = HistogramComparison(1, 0.8, cv2.HISTCMP_CORREL);
    # Create a new shuffler
    s = BruteForceShuffler(matrix);
    
    best_match = -1;
    best_matrix = matrix;
    
    for iter in range (math.factorial(dim[0]*dim[1])):
        s.flatten();
        s.shuffleFlattened();
        pieces = s.unFlatten();
        
        vals = [];
        for i in range (dim[0]):
            for j in range (dim[1]):    
                temp = np.mean(hc.match(pieces, i, j));
                vals.append(temp);
        
        if (best_match < np.mean(vals)): 
            best_matrix = pieces;
            best_match = np.mean(vals);
            
    showComplete(best_matrix, image, 'Result');
    
    print (s.getMaxAttempts())
    

#for i in range (9):
#    image = cv2.imread("../Tiles_shuffled/tiles_shuffled_2x2_0" + str(i) + ".png");
#    dim = [2,2];
#    match2x2(image, dim);


def main():
    image = cv2.imread("../Tiles_shuffled/tiles_shuffled_3x3_05.png");
    dim = [3,3];
    match2x2(image, dim);
    
def flatten (puzzlePieces):
    dimx,dimy,r,c,d = puzzlePieces.shape;
    return puzzlePieces.reshape((dimx*dimy, r, c, d));

def showFlattened(flattened):
    """
    Plot the puzzle in flattened state.
    """
    l = len(flattened);
    im = flattened[0]
    for i in range (1, l):
        im = np.hstack((im, flattened[i]));
        
    plt.imshow(im);
    plt.show();
    
def unFlatten(flattened, dim):
    """
    Restore puzzle to original shape.
    """
    r, c, d = flattened[0].shape;
    im = np.zeros((dim[0], dim[1], r, c, d));
    
    for i in range (len(flattened)):
        x = i // dim[0];
        y = i % dim[0];
        im[x,y] = flattened[i];
    return im;


def match ():
    """ 
    Attempt at matching larger puzzles.
    """
    
    image = cv2.imread("../Tiles_shuffled/tiles_shuffled_5x5_07.png");
    dim = [5,5];    
    matrix = divideImage(image, dim);
    hc = HistogramComparison(3, 0.8, cv2.HISTCMP_CORREL);
    
    # Get initial flattened array
    flattened = flatten(matrix);
    
    # Set variables to keep score for the best current match
    result_value = 0;
    result = np.copy(flattened);

    # Go through all possible first pieces
    for index in range (len(flattened)):        
        f = np.copy(flattened);
        
#        indices = list(p[index]);
#        # Get the rest of the range to add to the list
#        r = list(range (dim[0]*dim[1]));
#        x = indices[0];
#        y = indices[1];
#        r.remove(x);
#        r.remove(y);
#        # Append to list
#        indices.extend(r);
#        f = f[indices];
#        showFlattened(f);  
        
        # Keep record of matched pieces and pieces that have not yet been matched
        matched = [f[index]];
        not_yet_matched = list(range(0,dim[0]*dim[1]));
        not_yet_matched.remove(index);
        match_value = 0;

        # Now go over the elements of each row
        for k in range (0, dim[0]):
            # Get best match for each piece of the row
            for i in range (0,dim[1]):
                # Keep record of best matches for comparison later
                best_val = 0;
                best_match = 0;
                if (len (not_yet_matched) > 0):
                    best_match = not_yet_matched[0];
                
                # Skip the first tile
                if (k == 0 and i == 0):
                    continue;
                
                # Check to make sure we stop at last element
                #if ((k*dim[0] + i) < (len(f)-1)):
                for j in range (len(not_yet_matched)):
                    temp = 0;

                    if (k > 0 and i == 0):
                        temp = hc.matchNeighbour(matched[(k-1)*dim[1]], f[not_yet_matched[j]], ps.DIR_DOWN);
#                            im = np.vstack((matched[(k-1)*dim[1]], f[not_yet_matched[j]]))
#                            plt.imshow(im);
#                            plt.show();
#                            print (temp)
                        #print (k, i, "Matched down", not_yet_matched[j])
                    elif (i > 0):
                        temp = hc.matchNeighbour(matched[-1], f[not_yet_matched[j]], ps.DIR_RIGHT);
                    elif (i < 1 and k == 0):
                        temp = hc.matchNeighbour(matched[-1], f[not_yet_matched[j]], ps.DIR_RIGHT);
                        
                    if (temp > best_val):
                        best_val = temp;
                        #print (k,i, ": replaced " + str(best_match) + " with " + str(not_yet_matched[j]));
                        best_match = not_yet_matched[j];
                
                # While there's pieces to add, add them...
                if (len(not_yet_matched) > 0):
                    matched.append(f[best_match]);
                    not_yet_matched.remove(best_match);
                    match_value = match_value + best_val;
                    
        # Show the match that was made and add the value to the list
        #showFlattened(f)
        #showFlattened(matched);
        #print (match_value)
        if (match_value > result_value):
            result_value = match_value;
            result = matched;
        
    # Show the final result in flattened form
    #showFlattened(result);
    
    # Show the final result in unflattened form
    u = unFlatten(result, dim);
    #showComplete(u, image, "Result");
                
print(timeit.timeit(match, number = 10));

        

    
