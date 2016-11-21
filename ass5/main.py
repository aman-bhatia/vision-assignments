import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne
import cv2

train_dir = 'train/'
test_dir = 'test/'
scale = 3
input_size = 33
label_size = 21
padding = (input_size - label_size)/2
stride = 14


def load_dataset():
    X_train = []
    y_train = []
    images = os.listdir(train_dir)
    for f in images:
        img = cv2.imread(train_dir + f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)[:,:,0]
        img = cv2.normalize(img.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        h,w = img.shape
        for i in xrange(0,h-input_size,stride):
            for j in xrange(0,w-input_size,stride):
                hres = img[i:i+input_size, j:j+input_size]
                lres = cv2.resize(cv2.resize(hres,None,fx=1.0/scale,fy=1.0/scale), None, fx=scale,fy=scale, interpolation=cv2.INTER_CUBIC)
                hres = hres[padding:padding+label_size, padding:padding+label_size]
                X_train.append(np.array([lres]))
                y_train.append(np.array([hres]))

    X_test = []
    y_test = []
    images = os.listdir(test_dir)
    for f in images:
        img = cv2.imread(test_dir + f)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)[:,:,0]
        img = cv2.normalize(img.astype('float32'), None, 0.0, 1.0, cv2.NORM_MINMAX)
        h,w = img.shape
        for i in xrange(0,h-input_size,stride):
            for j in xrange(0,w-input_size,stride):
                hres = img[i:i+input_size, j:j+input_size]
                lres = cv2.resize(cv2.resize(hres,None,fx=1.0/scale,fy=1.0/scale), None, fx=scale,fy=scale, interpolation=cv2.INTER_CUBIC)
                hres = hres[padding:padding+label_size, padding:padding+label_size]
                X_test.append(np.array([lres]))
                y_test.append(np.array([hres]))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test


def build_cnn(input_var=None):

    network = lasagne.layers.InputLayer(shape=(None, 1, 33, 33),
                                        input_var=input_var)

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(9, 9),
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.ParametricRectifierLayer(network, alpha=lasagne.init.Constant(0))

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(1, 1),
            W=lasagne.init.GlorotUniform())

    network = lasagne.layers.ParametricRectifierLayer(network, alpha=lasagne.init.Constant(0))

    network = lasagne.layers.Conv2DLayer(
            network, num_filters=1, filter_size=(5, 5),
            W=lasagne.init.GlorotUniform())

    # network = lasagne.layers.DenseLayer(network,
    #         num_units=33*33,
    #         nonlinearity=lasagne.nonlinearities.softmax)

    # network = lasagne.layers.ReshapeLayer(network, shape=(33*33,))

    return network



def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]



def main(num_epochs=500):
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_test, y_test = load_dataset()
    print X_train.shape
    print y_train.shape
    print X_test.shape
    print y_test.shape

    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.tensor4('targets')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_cnn(input_var)
    
    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    prediction = lasagne.layers.get_output(network)
    loss = lasagne.objectives.squared_error(prediction, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.0001, momentum=0.9)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.squared_error(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    # Compile a function performing a training step on a mini-batch (by giving
    # the updates dictionary) and returning the corresponding training loss:
    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
    for epoch in range(num_epochs):
        # In each epoch, we do a full pass over the training data:
        train_err = 0
        train_batches = 0
        start_time = time.time()
        # i=1
        for batch in iterate_minibatches(X_train, y_train, 128, shuffle=True):
            # print i
            # i += 1
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, num_epochs, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))

    # After training, we compute and print the test error:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, 128, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))



main()