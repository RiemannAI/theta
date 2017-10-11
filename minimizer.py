#!/usr/bin/env sage
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from cma import fmin, CMAEvolutionStrategy
import numpy as np
import multiprocessing as mp


class Minimizer(object):
    """Abstract class layer for the implementation of minimizers"""
    __metaclass__ = ABCMeta

    def __init__(self):
        self._rtbm = None
        self._data = None

    @abstractmethod
    def train(self, rtbm, data, initial_solution=None):
        pass

    def cost(self, params):
        func = self._rtbm.copy()
        func.assign_params(params)
        try:
            res = -np.sum(np.log(func(self._data)))
            if np.isnan(res): res = np.inf
        except:
            res = np.inf
        return res


class CMA(Minimizer):
    """Implements the GA using CMA library"""
    def __init__(self, multi_thread=False):
        super(CMA, self).__init__()
        if multi_thread:
            self.num_cores = mp.cpu_count()
        else:
            self.num_cores = 1
        print('CMA on %d cpu(s) enabled' % self.num_cores)

    def train(self, rtbm, data, initial_solution=None):
        """The training algorithm"""

        # send data to parent class where cost function is defined
        self._data = data
        self._rtbm = rtbm

        bmin, bmax = rtbm.get_bounds()
        args = {'bounds': [bmin, bmax]}
        sigma = np.max(bmax)*0.1
        initsol = rtbm.size()*[1e-5]
        if initial_solution is not None:
            initsol = initial_solution

        pool = mp.Pool(self.num_cores)
        es = CMAEvolutionStrategy(initsol, sigma, args)
        while not es.stop():
            solutions = es.ask()
            f_values = pool.map_async(self.cost, solutions).get()
            es.tell(solutions, f_values)
            es.logger.add()
            es.disp()
        print(es.result)

        rtbm.assign_params(es.result[0])
        return es.result[0]

    @property
    def num_cores(self):
        return self._num_cores

    @num_cores.setter
    def num_cores(self, cores):
        if cores > mp.cpu_count():
            print('CMA: the number of requested CPU is larger than cpu_count.')
        elif cores <= 0:
            raise AssertionError('CMA: the requested number of CPU is <= 0')
        self._num_cores = cores

