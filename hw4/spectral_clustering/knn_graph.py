import numpy as np
import scipy.spatial.distance

def knn_graph(X, k, threshold):
    '''
    KNN_GRAPH Construct W using KNN graph

        Input:
            X - data point features, n-by-p maxtirx.
            k - number of nn.
            threshold - distance threshold.

        Output:
            W - adjacency matrix, n-by-n matrix.
    '''

    # YOUR CODE HERE
    # begin answer
    N,P = X.shape
    W = np.zeros((N,N))
    dist = scipy.spatial.distance.cdist(X, X)
    sorted_dist = np.argsort(dist,axis=1)
    
    for i in range(N):
        for j in range(k):
            if dist[i][sorted_dist[i][j]] < threshold and i != sorted_dist[i][j]:
                W[i][sorted_dist[i][j]] = 1
                W[sorted_dist[i][j]][i] = 1
            
    return W
    # end answer
