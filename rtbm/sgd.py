# -*- coding: utf-8 -*-
from __future__ import print_function

import time
import numpy as np
import matplotlib.pyplot as plt


def train(cost, model, input_x_data, input_y_data, validation_split, validation_x_data, validation_y_data, stopping,
          scheme, maxiter, batch_size, shuffle, lr, decay, momentum,nesterov, noise, cplot):
    """Trains the given model with stochastic gradient descent methods

    :param cost: the cost fuction class
    :param model: the model to be trained
    :param x_data: the target data support
    :param y_data: the target data prediction
    :param scheme: the SGD method (Ada, RMSprop, see gradientschemes.py)
    :param validation_split: fraction of data used for validation only
    :param validation_x_data: external set of validation support
    :param validation_y_data: external set of validation target
    :param stopping: the stopping class (see stopping.py)
    :param maxiter: maximum number of allowed iterations
    :param batch_size: the batch size
    :param shuffle : shuffle the data on each iteration
    :param lr: learning rate
    :param decay: learning rate decay rate
    :param momentum: add momentum
    :param nesterov: add nesterov momentum
    :param noise: add gaussian noise
    :param cplot: if True shows the cost function evolution
    :return: dictionary with iterations and cost functions
    """

    # verify x and y data have the same length
    if input_y_data is not None:
        assert input_x_data.shape == input_y_data.shape, 'input x_data and y_data shape does not match'

    # check if validation_split and validation_x_data are set simultaneously
    if validation_split > 0 and validation_x_data is not None:
        raise AssertionError('validation_split and validation_x_data cannot be used simultaneously')

    # if validation_y_data is passed, check it matches the shape of its support
    if validation_y_data is not None:
        assert validation_x_data.shape == validation_y_data.shape, 'validation x and y data shapes do not match'

    # verify that validation_split doesn't kill all training points
    assert validation_split < 1, 'validation_split too large, no training data'

    # prepare trainng and validation data
    training_x_data = input_x_data
    training_y_data = input_y_data

    # create validation set
    if validation_split > 0:
        size = int(input_x_data.shape[1] * validation_split)
        training_x_data = input_x_data[:,:-size]
        validation_x_data = input_x_data[:,input_x_data.shape[1]-size:]
        if input_y_data is not None:
            training_y_data = input_y_data[:, :-size]
            validation_y_data = input_y_data[:,input_y_data.shape[1]-size:]
        print('Split summary: training size %d | validation size %d' % (training_x_data.shape[1], size))

    # Generate batches
    RE = 0
    if batch_size > 0:
        BS = training_x_data.shape[1] / batch_size
        if training_x_data.shape[1] % batch_size > 0:
            RE = 1
    else:
        BS = 1
        batch_size = training_x_data.shape[1]

    # Switch on/off noise
    nF = 0
    if noise > 0:
        nF = 1

    t0 = time.time()

    cost_tr_hist = np.zeros(maxiter)
    cost_val_hist = np.zeros(maxiter)
    
    # Get inital W parameter
    W = model.get_parameters()
    oldG = np.zeros(W.shape)

    # Loop over epoches
    stop_iteration = maxiter
    shuffled_indexes = np.arange(training_x_data.shape[1])
    for i in range(maxiter):

        if shuffle:
            np.random.shuffle(shuffled_indexes)
        
        shuffled_x_data = training_x_data[:, shuffled_indexes]
        shuffled_y_data = None

        if training_y_data is not None:
            shuffled_y_data = training_y_data[:, shuffled_indexes]

        partial_cost_tr_batch = 0
        partial_cost_val_batch = 0

        # Loop over batches
        for b in range(BS+RE):
            
            # Prepare data    
            data_x = shuffled_x_data[:,b*batch_size:(b+1)*batch_size]
            data_y = None
            if shuffled_y_data is not None:
                data_y = shuffled_y_data[:,b*batch_size:(b+1)*batch_size]

            # Feedforward
            Xout = model.feed_through(data_x, True)
            
            # Calc cost for training and validation
            partial_cost_tr_batch += cost.cost(Xout, data_y)
            if validation_x_data is not None:
                Xval = model.feed_through(validation_x_data, False)
                partial_cost_val_batch += cost.cost(Xval, validation_y_data)

            # Backprop
            model.backprop(cost.gradient(Xout, data_y))

            # Get gradients
            G = model.get_gradients()
         
            if scheme is not None:
                G = scheme.getupdate(G, lr)
            else:
                G = lr*G

            # Adjust weights (with momentum)
            if momentum != 0:
                G = G + momentum*oldG 
                oldG = G
            
            # Set new weights
            if nF == 0:
                W = W - G
            else:
                W = W - G - np.random.normal(0, lr/(1+i)**noise, oldG.shape)
        
            if nesterov:
                model.set_parameters(W-momentum*oldG) # Nesterov update
            else:    
                model.set_parameters(W)
            
        # Decay learning rate
        lr = lr*(1-decay)

        # fill cost histogram
        cost_tr_hist[i] = partial_cost_tr_batch/(BS+RE)
        cost_val_hist[i] = partial_cost_val_batch/(BS+RE)

        # print to screen
        progress_bar(i+1, maxiter, suffix="| iteration %d in %.2f(s) | cost = %f | val = %f" % (i+1, time.time()-t0, cost_tr_hist[i], cost_val_hist[i]))

        # stop condition
        if stopping is not None:
            if stopping.do_stop(cost_val_hist[:i]):
                print('\nStopping condition achieved at iteration %d' % (i+1))
                stop_iteration = i + 1
                break

    I = np.linspace(1, maxiter, maxiter)
    if cplot:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,2))
        ax1.plot(I, cost_tr_hist, '-', label='training')
        ax1.set_ylabel("C", rotation=0, labelpad=10)
        ax1.axvline(stop_iteration, c='g', label='stop iteration')
        ax1.legend()
        ax2.plot(I, cost_val_hist, 'r-', label='validation')
        ax2.axvline(stop_iteration, c='g', label='stop iteration')
        ax2.legend()

    return {'iterations': I, 'cost_tr': cost_tr_hist, 'cost_val': cost_val_hist}


def progress_bar(iteration, total, suffix='', length=20, fill='█'):
        """Call in a loop to create terminal progress bar
        Args:
            iteration   - Required  : current iteration (Int)
            total       - Required  : total iterations (Int)
            suffix      - Optional  : suffix string (Str)
            length      - Optional  : character length of bar (Int)
            fill        - Optional  : bar fill character (Str)
        """
        percent = ("{0:." + str(1) + "f}").format(100 * (iteration / float(total)))
        filled = int(length * iteration // total)
        bar = fill * filled + '-' * (length - filled)
        print('\rProgress: |%s| %s%% %s' % (bar, percent, suffix), end='\r')
        
        if iteration == total:
            print()

   