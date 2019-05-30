import numpy as np
import math

def logistic(X, y1):
    '''
    LR Logistic Regression.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.random.rand(P + 1, 1)
    
    # YOUR CODE HERE
    # begin answer
    y = np.zeros([1,N])
    for i in range(y1.shape[1]):
        if y1[0][i]>0:
            y[0][i]=1
    X = np.vstack((np.ones(N).T,X))
    loss = -1./N*(np.dot(np.dot(w.T,X),y.T)-np.sum(np.log(1+np.exp(np.dot(w.T,X)))))
    lamda = 0.02
    iters = 0

    while loss.all()>0 and iters < 5000:
        gradient = -1./N*(np.dot(X,y.T)-np.dot(X,(1./(1+np.exp(np.dot(-1*w.T,X)))).T))
        w = w - lamda * gradient
        loss = -1./N*(np.dot(np.dot(w.T,X),y.T)-np.sum(np.log(1+np.exp(np.dot(w.T,X)))))

        iters += 1
       
    
    # end answer
    
    return w
