# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import numpy as np


class initializer(object):
    """ Abstract class for cost functions """
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def getinit(self, S):
        pass
    
class uniform(initializer):
    
    def __init__(self, bound=1, center=0):
        self._bound = bound
        self._center = center
    
    def getinit(self, S):
        return np.random.uniform(-self._bound, self._bound, S)+self._center
    
class normal(initializer):
    def __init__(self, mean=0, sdev=1):
        self._mean = mean
        self._sdev = sdev
                
    def getinit(self, S):
        
        return np.random.normal(self._mean, self._sdev, S)
    
class null(initializer):
    
    def getinit(self, S):
        
        return np.zeros(S)    
    
class glorot_normal(initializer):
    def getinit(self, S):
        return np.random.normal(0, 2.0/(S[0]+S[1]),S)
    
class glorot_uniform(initializer):
    def getinit(self, S):
        limit = np.sqrt(6.0/(S[0]+S[1]))
                        
        return np.random.uniform(-limit,limit,S)