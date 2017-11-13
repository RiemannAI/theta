# -*- coding: utf-8 -*-
from __future__ import print_function

import time
import numpy as np
import matplotlib.pyplot as plt


def train(cost, model, x_data, y_data, scheme, maxiter, batch_size,
          lr, decay, momentum,nesterov, noise, cplot):
    """Trains the given model with stochastic gradient descent methods

    :param cost: the cost fuction class
    :param model: the model to be trained
    :param x_data: the target data support
    :param y_data: the target data prediction
    :param scheme: the SGD method (Ada, RMSprop, see gradientschemes.py)
    :param maxiter: maximum number of allowed iterations
    :param batch_size: the batch size
    :param lr: learning rate
    :param decay: learning rate decay rate
    :param momentum: add momentum
    :param nesterov: add nesterov momentum
    :param noise: add gaussian noise
    :param cplot: if True shows the cost function evolution
    :return: dictionary with iterations and cost functions
    """

    
    # Generate batches
    RE = 0
    if batch_size > 0:
        BS = x_data.shape[1] / batch_size
        if(x_data.shape[1] % batch_size > 0):
            RE = 1
    else:
        BS = 1
        batch_size = x_data.shape[1]

    # Switch on/off noise
    nF = 0
    if noise > 0 :
        nF = 1

    t0 = time.time()

    cost_hist = np.zeros(maxiter)
    
    # Get inital W parameter
    W = model.get_parameters()
    oldG = np.zeros(W.shape)

    # Loop over epoches
    shuffled_indexes = np.arange(x_data.shape[1])
    for i in range(0, maxiter):

        np.random.shuffle(shuffled_indexes)
        train_data_x = x_data[:, shuffled_indexes]
        train_data_y = y_data[:, shuffled_indexes]

        # Loop over batches
        for b in range(0, BS+RE):
            
            # Prepare data    
            data_x = train_data_x[:,b*batch_size:(b+1)*batch_size]
            data_y = train_data_y[:,b*batch_size:(b+1)*batch_size]
            
            # Feedforward
            Xout = model.feed_through(data_x, True)
            
            # Calc cost
            cost_hist[i] = cost.cost(Xout,data_y)
            
            # Backprop
            model.backprop(cost.gradient(Xout,data_y))

            # Get gradients
            G = model.get_gradients()
         
            if scheme is not None:
                G = scheme.getupdate(G, lr)
            else:
                G = lr*G

            # Adjust weights (with momentum)
            if(momentum!=0):
                G = G + momentum*oldG 
                oldG = G
            
            # Set new weights
            if(nF == 0):
                W = W - G
            else:
                W = W - G - np.random.normal(0, lr/(1+i)**noise, oldG.shape)
          
        
            if(nesterov==True):
                # Nesterov update
                 model.set_parameters(W-momentum*oldG)
            else:    
                model.set_parameters(W)
            
            
            
        # Decay learning rate
        lr = lr*(1-decay)

        # print to screen
        progress_bar(i+1, maxiter, suffix="| iteration %d in %.2f(s) | cost = %f" % (i+1, time.time()-t0, cost_hist[i]))

    I = (np.linspace(0, maxiter-1, maxiter))
    if cplot:
        plt.figure(figsize=(3,2))
        plt.plot(I, cost_hist,"-")
        plt.ylabel("C", rotation=0, labelpad=10) 
        plt.show()
        
    return {'iterations': I, 'cost': cost_hist}



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

   