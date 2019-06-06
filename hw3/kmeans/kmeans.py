import numpy as np
import copy
from scipy.spatial.distance import cdist


def kmeans(x, k):
    '''
    KMEANS K-Means clustering algorithm

        Input:  x - data point features, n-by-p maxtirx.
                k - the number of clusters

        OUTPUT: idx  - cluster label, n-by-1
                ctrs - cluster centers, K-by-p matrix.
                iter_ctrs - cluster centers of each iteration, (iter, k, p)
                        3D matrix.
    '''
    # YOUR CODE HERE

    # begin answer
    
    max_iter = 1000
    iters = 1
    iter_ctrs = []

    ctrs = x[np.random.choice(x.shape[0],k,replace=False),:]
    iter_ctrs.append(ctrs)
    dist = cdist(x, ctrs)
    idx = np.argsort(dist,axis=1)[:,0]
    idx_last = idx
    
    for i in range(k):
        idx1 = np.where(idx==i)   
        ctrs[i] = np.mean(x[idx1],axis=0)
    iter_ctrs.append(copy.deepcopy(ctrs))
    dist = cdist(x,ctrs)
    idx = np.argsort(dist,axis=1)[:,0]
    
    while iters < max_iter and not np.array_equal(idx,idx_last):
        
        for i in range(k):
            idx1 = np.where(idx==i)  
            ctrs[i] = np.mean(x[idx1],axis=0)
        idx_last = idx
        iter_ctrs.append(copy.deepcopy(ctrs))
        dist = cdist(x,ctrs)
        idx = np.argsort(dist,axis=1)[:,0]
        
        iters += 1
    
    iter_ctrs.append(copy.deepcopy(ctrs))
    
    iter_ctrs = np.array(iter_ctrs)
    
    # end answer

    return idx, ctrs, iter_ctrs
