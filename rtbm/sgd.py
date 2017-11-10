# -*- coding: utf-8 -*-
from __future__ import print_function

import time
import numpy as np
import matplotlib.pyplot as plt


def train(cost, model, x_data, y_data, maxiter, batch_size,
          lr, decay, momentum,nesterov, noise, cplot):

    oldG = np.zeros(model.get_parameters().shape)

    # Generate batches
    RE = 0
    if(batch_size > 0):
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
    
    # Loop over epoches
    for i in range(0, maxiter):

        # Loop over batches
        for b in range(0, BS+RE):

            data_x = x_data[:,b*batch_size:(b+1)*batch_size]
            data_y = y_data[:,b*batch_size:(b+1)*batch_size]

            Xout = model.feed_through(data_x, True)
            cost_hist[i] = cost.cost(Xout,data_y)
            model.backprop(cost.gradient(Xout,data_y))

            W = model.get_parameters()

            # Nesterov update
            if(nesterov==True):
                model.set_parameters(W-momentum*oldG)

            # Get gradients
            G = model.get_gradients()

            # Adjust weights (with momentum)
            U = lr*G + momentum*oldG + nF*np.random.normal(0, lr/(1+i)**noise, oldG.shape)
            oldG = U

            W = W - U

            # Set gradients
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

   