import numpy as np
from kmeans import kmeans

def spectral(W, k):
    '''
    SPECTRUAL spectral clustering

        Input:
            W: Adjacency matrix, N-by-N matrix
            k: number of clusters

        Output:
            idx: data point cluster labels, n-by-1 vector.
    '''
    # YOUR CODE HERE
    # begin answer
    N = W.shape[0]
    idx = np.zeros(N)
    
    D = np.diag(np.sum(W,axis=1))
    L = D - W
  
    vals, vecs = np.linalg.eig(L)
    
 
    vecs = vecs[:,np.argsort(vals)]
    idx = kmeans(vecs[:,1:k],k)

    return idx
    
    
    # end answer
