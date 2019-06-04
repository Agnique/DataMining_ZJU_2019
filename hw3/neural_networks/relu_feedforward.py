import numpy as np

def relu_feedforward(in_):
    '''
    The feedward process of relu
      in_:
              in_: the input, shape: any shape of matrix
      
      outputs:
              out : the output, shape: same as in
    '''
    # TODO
    # f(x) = max(0,x)
    # begin answer
    out = np.zeros((in_.shape[0],in_.shape[1]))
    for i in range(in_.shape[0]):
        for j in range(in_.shape[1]):
            if in_[i][j] > 0:
                out[i][j] = in_[i][j]
    # end answer
    return out
