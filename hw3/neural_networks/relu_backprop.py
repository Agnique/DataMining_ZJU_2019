import numpy as np

def relu_backprop(in_sensitivity, in_):
    '''
    The backpropagation process of relu
      input paramter:
          in_sensitivity  : the sensitivity from the upper layer, shape: 
                          : [number of images, number of outputs in feedforward]
          in_             : the input in feedforward process, shape: same as in_sensitivity
      
      output paramter:
          out_sensitivity : the sensitivity to the lower layer, shape: same as in_sensitivity
    '''
    # TODO

    # begin answer
    num_img,num_out = in_sensitivity.shape
    out_sensitivity = np.zeros((num_img,num_out))
    for i in range(num_img):
        for j in range(num_out):
            if in_[i][j] > 0:
                out_sensitivity[i][j] = in_sensitivity[i][j]
    # end answer
    return out_sensitivity

