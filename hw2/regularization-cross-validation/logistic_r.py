import numpy as np

def logistic_r(X, y, lmbda):
    '''
    LR Logistic Regression.

      INPUT:  X:   training sample features, P-by-N matrix.
              y:   training sample labels, 1-by-N row vector.
              lmbda: regularization parameter.

      OUTPUT: w    : learned parameters, (P+1)-by-1 column vector.
    '''
    P, N = X.shape
    w = np.random.rand(P + 1, 1)
    # YOUR CODE HERE
    # begin answer
    for i in range(y.shape[1]):
        if y[0][i] < 0:
            y[0][i] = 0
    
    X = np.vstack((np.ones(N).T,X))
    loss = -1./N*(np.dot(np.dot(w.T,X),y.T)-np.sum(np.log(1+np.exp(np.dot(w.T,X))))) + 0.5 / N * lmbda * np.dot(w.T,w)
    iters = 0
    lamda = 2
    
    while loss.all()>0 and iters < 10000:
        gradient = -1./N*(np.dot(X,y.T)-np.dot(X,(1./(1+np.exp(np.dot(-1*w.T,X)))).T)) + 1./N* lmbda * w
        w = w - lamda*gradient
        loss = -1./N*(np.dot(np.dot(w.T,X),y.T)-np.sum(np.log(1+np.exp(np.dot(w.T,X))))) + 0.5 / N * lmbda * np.dot(w.T,w)
        #print loss
        iters += 1
    # end answer
    #print 'roud over'
    return w
