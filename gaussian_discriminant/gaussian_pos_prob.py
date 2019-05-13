import numpy as np
import math

def gaussian_pos_prob(X, Mu, Sigma, Phi):
    '''
    GAUSSIAN_POS_PROB Posterior probability of GDA.
    Compute the posterior probability of given N data points X
    using Gaussian Discriminant Analysis where the K gaussian distributions
    are specified by Mu, Sigma and Phi.
    Inputs:
        'X'     - M-by-N numpy array, N data points of dimension M.
        'Mu'    - M-by-K numpy array, mean of K Gaussian distributions.
        'Sigma' - M-by-M-by-K  numpy array (yes, a 3D matrix), variance matrix of
                  K Gaussian distributions.
        'Phi'   - 1-by-K  numpy array, prior of K Gaussian distributions.
    Outputs:
        'p'     - N-by-K  numpy array, posterior probability of N data points
                with in K Gaussian distribsubplots_adjustutions.
    ''' 
    N = X.shape[1]
    K = Phi.shape[0]
    p = np.zeros((N, K))
    #Your code HERE

    # begin answer
    
    l = np.zeros((N,K)) # P(x=i|y=j)
    for i in range(N):
        px = 0
        for j in range(K):
            l[i,j] = 1/(2*math.pi*math.sqrt(np.linalg.det(Sigma[:,:,j])))*math.exp(-1/2*np.dot(np.dot((X[:,i]-Mu[:,j]).transpose(),np.linalg.inv(Sigma[:,:,j])),(X[:,i]-Mu[:,j])))
            px += l[i,j]*Phi[j]
        p[i,j] = l[i,j]*Phi[j]/px
            
    
    # end answer
    
    
    
    return p
    