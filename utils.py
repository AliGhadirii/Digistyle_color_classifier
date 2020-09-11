from os import listdir
from matplotlib import image
import numpy as np
import math


def load_dataset():

    loaded_images = list()
    loaded_images_test = list()
    y_train = list()
    y_test = list()
    for filename in listdir('pink'):
        # load image
        img_data = image.imread('pink/' + filename)
        y_train.append(0)
        # store loaded image
        loaded_images.append(img_data)
    for filename in listdir('purple'):
        # load image
        img_data = image.imread('purple/' + filename)
        y_train.append(1)
        # store loaded image
        loaded_images.append(img_data)
    for filename in listdir('yellow'):
        # load image
        img_data = image.imread('yellow/' + filename)
        y_train.append(2)
        # store loaded image
        loaded_images.append(img_data)
    for filename in listdir('orange'):
        # load image
        img_data = image.imread('orange/' + filename)
        y_train.append(3)
        # store loaded image
        loaded_images.append(img_data)
    for filename in listdir('white'):
        # load image
        img_data = image.imread('white/' + filename)
        y_train.append(4)
        # store loaded image
        loaded_images.append(img_data)
    for filename in listdir('silver'):
        # load image
        img_data = image.imread('silver/' + filename)
        y_train.append(5)
        # store loaded image
        loaded_images.append(img_data)
    for filename in listdir('grey'):
        # load image
        img_data = image.imread('grey/' + filename)
        y_train.append(6)
        # store loaded image
        loaded_images.append(img_data)
    for filename in listdir('black'):
        # load image
        img_data = image.imread('black/' + filename)
        y_train.append(7)
        # store loaded image
        loaded_images.append(img_data)
    for filename in listdir('red'):
        # load image
        img_data = image.imread('red/' + filename)
        y_train.append(8)
        # store loaded image
        loaded_images.append(img_data)
    for filename in listdir('brown'):
        # load image
        img_data = image.imread('brown/' + filename)
        y_train.append(9)
        # store loaded image
        loaded_images.append(img_data)
    for filename in listdir('green'):
        # load image
        img_data = image.imread('green/' + filename)
        y_train.append(10)
        # store loaded image
        loaded_images.append(img_data)
    for filename in listdir('blue'):
        # load image
        img_data = image.imread('blue/' + filename)
        y_train.append(11)
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

    classes = np.array([0,1,2,3,4,5,6,7,8,9,10,11])  # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def random_mini_batches(X, Y, mini_batch_size=64, seed=0):
    m = X.shape[1]  # number of training examples
    mini_batches = []
    np.random.seed(seed)

    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0], m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(
        m / mini_batch_size)  # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size: k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size: m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)

    return mini_batches