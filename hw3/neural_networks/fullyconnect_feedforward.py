import numpy as np

def fullyconnect_feedforward(in_, weight, bias):
    '''
    The feedward process of fullyconnect
      input parameters:
          in_     : the intputs, shape: [number of images, number of inputs]
          weight  : the weight matrix, shape: [number of inputs, number of outputs]
          bias    : the bias, shape: [number of outputs, 1]

      output parameters:
          out     : the output of this layer, shape: [number of images, number of outputs]
    '''
    # TODO

    # begin answer
    N_img, N_in = in_.shape
    out = np.dot(in_, weight) + np.tile(bias.T,(N_img,1))
    
    # end answer

    return out

