import numpy as np
import scipy.stats
from scipy.spatial.distance import cdist

def knn(x, x_train, y_train, k):
    '''
    KNN k-Nearest Neighbors Algorithm.

        INPUT:  x:         testing sample features, (N_test, P) matrix.
                x_train:   training sample features, (N, P) matrix.
                y_train:   training sample labels, (N, ) column vector.
                k:         the k in k-Nearest Neighbors

        OUTPUT: y    : predicted labels, (N_test, ) column vector.
    '''

    # Warning: uint8 matrix multiply uint8 matrix may cause overflow, take care
    # Hint: You may find numpy.argsort & scipy.stats.mode helpful

    # YOUR CODE HERE

    # begin answer

    dists = cdist(x_train, x)
    idx = np.argpartition(dists, k, axis=0)[:k]
    neighbors = np.take(y_train,idx)
    y = scipy.stats.mode(neighbors,axis=0)[0]
    
    # end answer

    return y
