# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import numpy as np


class gradientscheme(object):
    """ Abstract class for cost functions """
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def getupdate(self, G):
        pass

    
class adagrad(gradientscheme):
    
    def __init__(self, epsilon=1e-5):
        self._eps = epsilon
        self._G2sum = np.empty(0)
        
    def getupdate(self, G, lr):
        if(self._G2sum.shape[0]==0):
            # Initalize
            self._G2sum = np.zeros(G.shape)
            self._Emat  = np.ones(G.shape)*self._eps
            
        # Accumulate gradients
        self._G2sum = self._G2sum + G**2
        
        return lr/np.sqrt(self._G2sum+self._Emat)*G
        
class RMSprop(gradientscheme):
    
    def __init__(self, rate=0.9, epsilon=1e-5):
        self._eps = epsilon
        self._E = np.empty(0)
        self._r = rate
        
    def getupdate(self, G, lr):
        if(self._E.shape[0]==0):
            # Initalize
            self._E = np.zeros(G.shape)
            self._Emat = np.ones(G.shape)*self._eps
            
        # Accumulate gradients
        self._E = self._r*self._E + (1-self._r)*G**2
        
        return lr/np.sqrt(self._E+self._Emat)*G
    
    
class adadelta(gradientscheme):
    def __init__(self, rate=0.9, epsilon=1e-5):
        self._eps = epsilon
        self._E = np.empty(0)
        self._Ed = np.empty(0)
        self._r = rate
    
    def getupdate(self, G, lr):
        if(self._E.shape[0]==0):
            # Initalize
            self._E  = np.zeros(G.shape)
            self._Ed = np.zeros(G.shape)
            
            self._Emat = np.ones(G.shape)*self._eps
            
        # Accumulate gradients
        self._E = self._r*self._E + (1-self._r)*G**2
        RMSg = np.sqrt(self._E+self._Emat)
        Dt = lr/RMSg*G
        
        R = np.sqrt(self._Ed+self._Emat)/RMSg
         
        self._Ed = self._r*self._Ed + (1-self._r)*(Dt**2)
      
        return R*G

class adam(gradientscheme):
    def __init__(self, b1=0.9, b2=0.999 ,epsilon=1e-8):
        self._eps = epsilon
        self._b1  = b1
        self._b2  = b2
        self._m   = np.empty(0)
        self._v    = np.empty(0)
        
    def getupdate(self, G, lr):
        if(self._m.shape[0]==0):
            # Initalize
            self._m  = np.zeros(G.shape)
            self._v = np.zeros(G.shape)
            
            self._Emat = np.ones(G.shape)*self._eps

        # Calc gradient    
        self._m = self._b1*self._m + (1-self._b1)*G
        self._v = self._b2*self._v + (1-self._b2)*(G**2)
            
        return lr*(self._m/(1-self._b1))/(np.sqrt(self._v/(1-self._b2))+self._Emat)