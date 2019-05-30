import numpy as np

def perceptron(X, y):
    '''
    PERCEPTRON Perceptron Learning Algorithm.

       INPUT:  X: training sample features, P-by-N matrix.
               y: training sample labels, 1-by-N row vector.

       OUTPUT: w:    learned perceptron parameters, (P+1)-by-1 column vector.
               iter: number of iterations

    '''
    P, N = X.shape
    w = np.ones((P + 1, 1))
    iters = 0
    D = np.zeros((1,N))
    # YOUR CODE HERE
    
    # begin answer
    flag = 0
    w = w.ravel()
    while iters < 1000 and flag == 0:
        iters = iters + 1
        D = np.dot(w[1:].T,X)*y + w[0]*y
        flag = 1
        for i in range(N):
            if D[0][i] <= 0:
                flag = 0
                w[1:] = w[1:] + X[:,i]*y[:,i]
                w[0] = w[0] + y[:,i]
       
    
    # end answer
    
    return w, iters