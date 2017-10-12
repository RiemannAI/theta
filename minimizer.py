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
        self._model = None
        self.X_data = None
        self.Y_data = None

    @abstractmethod
    def train(self, model, X_data, Y_data):
        self._model = model
        self.X_data = X_data
        self.Y_data = Y_data

    def set_costfunction(self, C):
        """ Sets the cost function to be used """
        self._costfunction = staticmethod(C)
        
        
    def cost(self, params):
        self._model.assign(params)
       
        res = self._costfunction.__func__(self._model(self.X_data),self.Y_data)
       
        return res

    
    

class CMA(Minimizer):
    """Implements the GA using CMA library"""
    def __init__(self):
        super(CMA, self).__init__()

    def train(self, model, X_data, Y_data):
        super(CMA, self).train(model, X_data, Y_data)
        bmin, bmax = model.get_bounds()
        sol = fmin(self.cost, model.size()*[1e-5], np.max(bmax)*0.1, {'bounds': [bmin, bmax]})
        model.assign(sol[0])
        return sol[0]


class DifferentialEvolution(Minimizer):
    """Implements the scipy optimize differential evolution"""
    def __init__(self):
        super(DifferentialEvolution, self).__init__()

    def train(self, model, X_data, Y_data):
        super(DifferentialEvolution, self).train(model, X_data, Y_data)
        bmin, bmax = model.get_bounds()
        sol = differential_evolution(self.cost, [ (bmin[i],bmax[i]) for i in range(len(bmin))], disp=True)
        model.assign(sol.x)
        return sol.x
