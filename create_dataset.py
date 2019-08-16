import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import cv2

path = './Hello//'

folder = glob(path + '*.jpg')


def create_dataset(imgs_loc, name, num_imgs, img_shape):
    """ The function returns a numpy array
    imgs_loc : location of the folder in which aall your images are placed
    name : name of the datafile ex. x_train or x_test
    num_imgs : Number of images to be added in the folder to be added in x_train or x_test
    img_shape : shape of the images for example in MNIST it is 28 x 28 then put 28"""
    name = np.array([])

    for i in range(num_imgs):
        img = plt.imread(imgs_loc[i])
        img = cv2.resize(img, (img_shape, img_shape))
        name = np.append(name, img)

    name = np.reshape(name, (num_imgs, img_shape, img_shape))
    return name


x_test = create_dataset(folder, x_train, 69, 28)
x_test = create_dataset(path, x_test, 28, 28)
