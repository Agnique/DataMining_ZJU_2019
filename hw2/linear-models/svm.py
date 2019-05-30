import numpy as np
from scipy.optimize import minimize

def svm(X, y):
    '''
    SVM Support vector machine.

    INPUT:  X: training sample features, P-by-N matrix.
            y: training sample labels, 1-by-N row vector.

    OUTPUT: w: learned perceptron parameters, (P+1)-by-1 column vector.
            num: number of support vectors

    '''
    P, N = X.shape
    w0 = np.zeros((P + 1, 1))
    w0 = w0.ravel()

    # YOUR CODE HERE
    # Please implement SVM with scipy.optimize. You should be able to implement
    # it within 20 lines of code. The optimization should converge wtih any method
    # that support constrain.
    # begin answer

    X = np.vstack((np.ones(N).T,X))
    def objective(w):
        return 0.5*np.sum(w*w)
    
    y = y.ravel()

    con = {'type':'ineq','fun': lambda w: y*np.sum(w*X.T,axis=1)-1.}
    sol = minimize(objective, w0, method='SLSQP', constraints=con)

    w = sol.x
    w = np.reshape(w,(P+1,1))
    d = y*np.dot(w.T,X)
    num = np.sum(np.absolute(d-1)<0.0001)

    # end answer
    return w, num

