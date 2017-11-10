# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import numpy as np
import sys

class actfunc(object):
    """ Abstract class for cost functions """
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def activation(self, X):
        pass
    
    @abstractmethod
    def gradient(self, X):
        pass
    
class linear(actfunc):
    """ linear pass through """
    
    @staticmethod
    def activation(x):
        return x
    @staticmethod
    def gradient(x):
        return np.ones(x.shape)
    
class sigmoid(actfunc):
    """ The sigmoid """

    @staticmethod
    def activation(x):   
        return 1.0/(1+np.exp(-x))
    
    @staticmethod
    def gradient(x):
        e = np.exp(x)
        return e/((1+e)**2)
    
class tanh(actfunc):
    """ The tanh """

    @staticmethod
    def activation(x):   
        return np.tanh(x)
    @staticmethod
    def gradient(x):
        return 1.0/(np.cosh(x)**2)

    
class softmax(actfunc):
    """ Softmax """
    
    @staticmethod
    def activation(x):
        E = np.exp(x)
        S = np.sum(E,axis=0) 
    
        return np.divide(E, S[np.newaxis,:])
       
    @staticmethod    
    def gradient(x):
        return 0