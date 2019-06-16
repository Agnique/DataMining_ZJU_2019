import numpy as np
import matplotlib.pyplot as plt
from pca import PCA
from scipy.ndimage import rotate

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def hack_pca(filename):
    '''
    Input: filename -- input image file name/path
    Output: img -- image without rotation
    '''
    img_r = (plt.imread(filename)).astype(np.float64)

    # YOUR CODE HERE
    # begin answer
    img_g = rgb2gray(img_r)
    x_idx, y_idx = np.where(img_g < 20)
    data = np.hstack((x_idx.reshape(-1, 1), y_idx.reshape(-1, 1)))
    eigvec, eigval = PCA(data)
    vec = eigvec[:,0]
    angle = np.arctan(vec[0]/vec[1]) * 180 / np.pi
    img = rotate(img_r, -angle).astype(np.int)
    
    return img
    # end answer