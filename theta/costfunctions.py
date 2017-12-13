# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import numpy as np
import sys


class costfunction(object):
    """ Abstract class for cost functions """
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def cost(self, X, *Y):
        pass
    
    @abstractmethod
    def gradient(self, X):
        pass
    

class mse(costfunction):
    """ Mean squared error """

    @staticmethod
    def cost(x, y):   
        return np.sum(np.mean((x-y)**2,axis=1))
    
    @staticmethod
    def gradient(x, y):
        return 2*(x-y)
    
        
class logarithmic(costfunction):
    """ Logarithmic total cost """
    
    @staticmethod
    def cost(x, *y):
        return -np.sum(np.log(x))

    @staticmethod
    def gradient(x, *y):
        return -1.0/x*x.shape[1]

        
class sum(costfunction):
    """ Sum total cost """
   
    @staticmethod
    def cost(x, *y):
        return -np.sum(x)

    @staticmethod
    def gradient(x):
        sys.exit("Gradient not implemented!")

        
class rmse(costfunction):
    """ Root mean squared error """
    
    @staticmethod
    def cost(x, y):
        return np.sqrt(np.sum(np.mean((y-x)**2,axis=1)))

    @staticmethod
    def gradient(x):
        return 1.0/np.sqrt(np.sum(np.mean((y-x)**2,axis=1)))*(x-y)
     
        
class crossentropy(costfunction):
    """ cross-entropy """
    
    @staticmethod
    def cost(x, y):
        lx  = np.log(x)
        return -np.sum(np.mean(np.multiply(y,lx),axis=1))
    
    @staticmethod
    def gradient(x):
        return -1.0/y.shape[1]*y/x
  
