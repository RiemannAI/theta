#!/usr/bin/env sage
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from cma import fmin
from scipy.optimize import differential_evolution
import numpy as np


class Minimizer(object):
    """Abstract class layer for the implementation of minimizers"""
    __metaclass__ = ABCMeta

    def __init__(self):
        self._rtbm = None
        self.X_data = None
        self.Y_data = None

    @abstractmethod
    def train(self, rtbm, X_data, Y_data):
        self._rtbm = rtbm
        self.X_data = X_data
        self.Y_data = Y_data

    def set_costfunction(C):
        """ Sets the cost function to be used """
        self._costfunction = staticmethod(C)
        
        
    def cost(self, params):
        self._rtbm.assign(params)
       
        res = self._costfunction.__func__(self._rtbm(self.X_data),Y_data)
       
        return res

    
    

class CMA(Minimizer):
    """Implements the GA using CMA library"""
    def __init__(self):
        super(CMA, self).__init__()

    def train(self, rtbm, data):
        super(CMA, self).train(rtbm, data)
        bmin, bmax = rtbm.get_bounds()
        sol = fmin(self.cost, rtbm.size()*[1e-5], np.max(bmax)*0.1, {'bounds': [bmin, bmax]})
        rtbm.assign(sol[0])
        return sol[0]


class DifferentialEvolution(Minimizer):
    """Implements the scipy optimize differential evolution"""
    def __init__(self):
        super(DifferentialEvolution, self).__init__()

    def train(self, rtbm, data):
        super(DifferentialEvolution, self).train(rtbm, data)
        bmin, bmax = rtbm.get_bounds()
        sol = differential_evolution(self.cost, [ (bmin[i],bmax[i]) for i in range(len(bmin))], disp=True)
        rtbm.assign(sol.x)
        return sol.x
