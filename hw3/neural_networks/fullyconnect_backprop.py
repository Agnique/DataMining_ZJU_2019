import numpy as np

def fullyconnect_backprop(in_sensitivity, in_, weight):
    '''
    The backpropagation process of fullyconnect
      input parameter:
          in_sensitivity  : the sensitivity from the upper layer, shape: 
                          : [number of images, number of outputs in feedforward]
          in_             : the input in feedforward process, shape: 
                          : [number of images, number of inputs in feedforward]
          weight          : the weight matrix of this layer, shape: 
                          : [number of inputs in feedforward, number of outputs in feedforward]

      output parameter:
          weight_grad     : the gradient of the weights, shape: 
                          : [number of inputs in feedforward, number of outputs in feedforward]
          out_sensitivity : the sensitivity to the lower layer, shape: 
                          : [number of images, number of inputs in feedforward]

    Note : remember to divide by number of images in the calculation of gradients.
    '''

    # TODO

    # begin answer
    num_img,num_out = in_sensitivity.shape
    num_img,num_in = in_.shape
    
    out_sensitivity = np.dot(in_sensitivity,weight.T)
    weight_grad = np.dot(in_.T,in_sensitivity) * 1.0 / num_img
    bias_grad = np.reshape(np.sum(in_sensitivity,axis=0),(num_out,1)) * 1.0 / num_img
    # end answer

    return weight_grad, bias_grad, out_sensitivity

