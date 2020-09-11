from os import listdir
from matplotlib import image
import numpy as np
import tensorflow as tf

def load_dataset():

    loaded_images = list()
    loaded_images_test = list()
    y_train = list()
    y_test = list()
    for filename in listdir('pink'):
        # load image
        img_data = image.imread('pink/' + filename)
        y_train.append(1)
        # store loaded image
        loaded_images.append(img_data)
    for filename in listdir('purple'):
        # load image
        img_data = image.imread('purple/' + filename)
        y_train.append(2)
        # store loaded image
        loaded_images.append(img_data)
    for filename in listdir('yellow'):
        # load image
        img_data = image.imread('yellow/' + filename)
        y_train.append(3)
        # store loaded image
        loaded_images.append(img_data)
    for filename in listdir('orange'):
        # load image
        img_data = image.imread('orange/' + filename)
        y_train.append(4)
        # store loaded image
        loaded_images.append(img_data)
    for filename in listdir('white'):
        # load image
        img_data = image.imread('white/' + filename)
        y_train.append(5)
        # store loaded image
        loaded_images.append(img_data)
    for filename in listdir('silver'):
        # load image
        img_data = image.imread('silver/' + filename)
        y_train.append(6)
        # store loaded image
        loaded_images.append(img_data)
    for filename in listdir('grey'):
        # load image
        img_data = image.imread('grey/' + filename)
        y_train.append(7)
        # store loaded image
        loaded_images.append(img_data)
    for filename in listdir('black'):
        # load image
        img_data = image.imread('black/' + filename)
        y_train.append(8)
        # store loaded image
        loaded_images.append(img_data)
    for filename in listdir('red'):
        # load image
        img_data = image.imread('red/' + filename)
        y_train.append(9)
        # store loaded image
        loaded_images.append(img_data)
    for filename in listdir('brown'):
        # load image
        img_data = image.imread('brown/' + filename)
        y_train.append(10)
        # store loaded image
        loaded_images.append(img_data)
    for filename in listdir('green'):
        # load image
        img_data = image.imread('green/' + filename)
        y_train.append(11)
        # store loaded image
        loaded_images.append(img_data)
    for filename in listdir('blue'):
        # load image
        img_data = image.imread('blue/' + filename)
        y_train.append(12)
        # store loaded image
        loaded_images.append(img_data)
    for filename in listdir('test'):
        # load image
        img_data = image.imread('test/' + filename)
        y_test.append(0)
        # store loaded image
        loaded_images_test.append(img_data)


    train_set_x_orig = np.array(loaded_images)  # your train set features
    train_set_y_orig = np.array(y_train)  # your train set labels


    test_set_x_orig = np.array(loaded_images_test)  # your test set features
    test_set_y_orig = np.array(y_test)  # your test set labels

    classes = np.array([1,2,3,4,5,6,7,8,9,10,11,12])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.

Y_train = convert_to_one_hot(Y_train_orig,12)
Y_test = convert_to_one_hot(Y_test_orig,12)

print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))