# -*- coding: utf-8 -*-

import numpy as np


class Model(object):
    
    def __init__(self):
        """ Initialize attributes """
        self._layers = []
        self._Np = 0
        self._lower_bounds = None
        self._upper_bounds = None
   
    def add(self, layer):
        """ Add layer to model """
        if len(self._layers) == 0 or self._layers[-1].get_Nout() == layer.get_Nin():
            # Add the layer
            self._layers.append(layer)
            
            # Increase the parameter counter
            self._Np = self._Np + layer.size()
        else: 
            print("Input of layer does not match output of previous layer! => Add ignored")

    def feed_through(self, X):
        """ Feeds the input X through all layers Vectorized """
        x = X
        for L in self._layers:
            x = L.feedin(x)
        return x

    def __call__(self, data):
        """ Evaluates the model for a given data array """
        return self.feed_through(data)

    def size(self):
        return self._Np

    def get_parameters(self):
        """ Collects all parameters and returns a flat array """
        R = []
        for L in self._layers:
            R.append(L.get_parameters())
            
        return np.concatenate(R)

    def get_bounds(self):
        """ Collects the bounds of the individual layers """
        A_L = []
        A_U = []
        
        for L in self._layers:
            bl, bu = L.get_bounds()
            A_L.append(bl)
            A_U.append(bu)
         
        self._lower_bounds = np.concatenate(A_L).tolist()
        self._upper_bounds = np.concatenate(A_U).tolist()

        return self._lower_bounds, self._upper_bounds

    def set_parameters(self, params):
        """ Set to the given parameters """
        Nt = 0
        
        for L in self._layers:
            
            L.set_parameters(params[Nt:Nt+L.size()])
            Nt += L.size()

    def predict(self, x):
        """ Performs prediction with the trained model """
        return self.feed_through(x)
