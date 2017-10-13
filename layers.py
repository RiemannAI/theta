#!/usr/bin/env sage
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np
from mathtools import factorized_hidden_expectation


class Layer(object):
    """ Abstract class for a layer of a deep network """
    __metaclass__ = ABCMeta

    @abstractmethod
    def feedin(self, X):
        pass
    
    @abstractmethod
    def get_parameters(self):
        pass
    
    @abstractmethod
    def set_parameters(self, params):
        pass
    
    @abstractmethod
    def size(self):
        pass
    
    @abstractmethod
    def get_bounds(self):
        pass

    def get_Nin(self):
        return self._Nin

    def get_Nout(self):
        return self._Nout


class ThetaUnitLayer(Layer):
    """ A layer of log-gradient theta units """

    def __init__(self, Nin, Nout, param_bound=10):
        self._Nin = Nin
        self._Nout = Nout

        # Parameter init
        self._bh = np.random.uniform(-1, 1,(Nout,1)).astype(complex)
        self._w = np.random.uniform(-1, 1,(Nin,Nout)).astype(complex)
        self._q = 10*np.diag(np.random.rand(Nout)).astype(complex)
     
        self._Np = 2*self._Nout+self._Nout*self._Nin
        
        # Generate allowed bounds
        self._param_bound = param_bound
        
    def feedin(self, X):
        """ Feeds in the data X and returns the output of the layer 
            Note: Vectorized 
        """

        return np.array(factorized_hidden_expectation(X,self._bh,self._w,self._q))

    def get_parameters(self):
        """ Returns the parameters as a flat array 
            [bh,w,q]
        """

        return np.concatenate([self._bh.flatten(),self._w.flatten(),self._q.diagonal()])

    def set_parameters(self, params):
        """ Set the matrices from flat input array P 
            P = [bh,w,q]
        """
        index = 0
        
        self._bh = params[index:index+self._bh.shape[0]].reshape(self._bh.shape)
        index += self._bh.shape[0]

        self._w = params[index:index+self._w.size].reshape(self._w.shape)
        index += self._w.size

        np.fill_diagonal(self._q, params[index:index+self._q.shape[0]])

    def size(self):
        """ Returns total # parameters """
        return self._Np
    
    def get_bounds(self):
        """Returns two arrays with min and max of each parameter for the GA"""
        self._lower_bounds = [-self._param_bound for _ in range(self._Np)]
        self._upper_bounds = [ self._param_bound for _ in range(self._Np)]

        # set q positive
        index = self._Np-self._q.shape[0]
        self._lower_bounds[index:] = [1E-5]*self._q.shape[0]

        return self._lower_bounds, self._upper_bounds 
    
