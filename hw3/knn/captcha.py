import urllib
import time
import os
import numpy as np
from extract_image import extract_image

def get_captcha():
    # get captcha
    for i in range(10):
        file_name = "captcha/" + str(i) + ".jpg"
        urllib.urlretrieve("http://jwbinfosys.zju.edu.cn/CheckCode.aspx", file_name)
        time.sleep(0.1)


def get_training_data():

    # get training data
    path = "captcha/"
    img_list = os.listdir(path)
    x_train = []
    y_train = []
    for img in img_list:
        x = extract_image("captcha/"+img)
        x = x.tolist()
        for i in range(5):
            x_train.append(x[i])
            y_train.append(int(img[i]))

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    np.savez("hack_data.npz",x_train = x_train,y_train = y_train)

