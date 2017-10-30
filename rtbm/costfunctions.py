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

    def cost(self, x, y):
        return 0.5*np.mean((x-y)**2)

    def gradient(self,X):
        return np.mean(x-y)
    
    
        
class logarithmic(costfunction):
    """ Logarithmic total cost """
    
    def cost(self, x, *y):
        return -np.sum(np.log(x))

    def gradient(self,X):
        sys.exit("Gradient not implemented!")

        
class sum(costfunction):
    """ Sum total cost """
   
    def cost(self, x, *y):
        return -np.sum(x)

    def gradient(self,X):
        sys.exit("Gradient not implemented!")

        
class rmse(costfunction):
    """ Root mean squared error """
    
    def cost(self, x, y):
        return np.sqrt(np.mean((x-y)**2))

    def gradient(self,X):
        sys.exit("Gradient not implemented!")

        
class crossentropy(costfunction):
    """ cross-entropy """
    
    def cost(self, x, y):
        lx  = np.log(x)
        lmx = np.log(1-x)
   
        return -np.mean(np.multiply(y,lx)+np.multiply(1-y,lmx))
    
    def gradient(self,X):
        sys.exit("Gradient not implemented!")

  
