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
        return np.sum(np.mean((x-y)**2,axis=1))
   
    def gradient(self, x, y):
        return 2*(x-y)
    
        
class logarithmic(costfunction):
    """ Logarithmic total cost """
    
    def cost(self, x, *y):
        return -np.sum(np.log(x))

    def gradient(self,x):
        sys.exit("Gradient not implemented!")

        
class sum(costfunction):
    """ Sum total cost """
   
    def cost(self, x, *y):
        return -np.sum(x)

    def gradient(self,x):
        sys.exit("Gradient not implemented!")

        
class rmse(costfunction):
    """ Root mean squared error """
    
    def cost(self, x, y):
        return np.sqrt(0.5*np.sum(np.mean((y-x)**2,axis=1)))

    def gradient(self,x):
        return 0.5/np.sqrt(0.5*np.sum(np.mean((y-x)**2,axis=1)))*(x-y)
     
        
class crossentropy(costfunction):
    """ cross-entropy """
    
    def cost(self, x, y):
        lx  = np.log(x)
        return -np.sum(np.mean(np.multiply(y,lx),axis=1))
    
    def gradient(self, x):
        return -1.0/y.shape[1]*y/x
  
