#!/usr/bin/env sage
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from cma import fmin, CMAEvolutionStrategy
from scipy.optimize import differential_evolution
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
        self._rtbm = rtbm
        self._data = data

    def cost(self, params):
        self._rtbm.assign_params(params)
        try:
            res = -np.sum(np.log(self._rtbm(self._data)))
            if np.isnan(res): res = np.inf
        except:
            res = np.inf
        return res


class CMA(Minimizer):
    """Implements the GA using CMA library"""
    def __init__(self, multi_thread=False):
        super(CMA, self).__init__()
        self._parallel = multi_thread
        if multi_thread:
            self._num_cores = mp.cpu_count()
            print('CMA multiprocessing on %d cpus enabled' % self._num_cores)

    def train(self, rtbm, data, initial_solution=None):
        super(CMA, self).train(rtbm, data)
        bmin, bmax = rtbm.get_bounds()
        args = {'bounds': [bmin, bmax]}
        sigma = np.max(bmax)*0.1
        initsol = rtbm.size()*[1e-5]
        if initial_solution is not None:
            initsol = initial_solution

        if self._parallel:
            es = CMAEvolutionStrategy(initsol, sigma, args)
            pool = mp.Pool(self._num_cores)
            while not es.stop():
                solutions = es.ask()
                f_values = pool.map_async(self.cost, solutions).get()
                es.tell(solutions, f_values)
                es.logger.add()
                es.disp()
            print(es.result)
            sol = es.result
        else:
            sol = fmin(self.cost, initsol, sigma, args)

        rtbm.assign_params(sol[0])
        return sol[0]

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


class DifferentialEvolution(Minimizer):
    """Implements the scipy optimize differential evolution"""
    def __init__(self):
        super(DifferentialEvolution, self).__init__()

    def train(self, rtbm, data, initial_solution=None):
        super(DifferentialEvolution, self).train(rtbm, data)
        bmin, bmax = rtbm.get_bounds()
        sol = differential_evolution(self.cost, [ (bmin[i],bmax[i]) for i in range(len(bmin))], disp=True)
        rtbm.assign_params(sol.x)
        return sol.x
