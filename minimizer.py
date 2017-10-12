#!/usr/bin/env sage
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from cma import CMAEvolutionStrategy
import numpy as np
import multiprocessing as mp
from contextlib import closing


class Minimizer(object):
    """Abstract class layer for the implementation of minimizers"""
    __metaclass__ = ABCMeta

    def __init__(self):
        self._cost_function = None
        self._model = None
        self.x_data = None
        self.y_data = None

    @abstractmethod
    def train(self, cost, model, x_data, y_data):
        self._cost_function = cost
        self._model = model
        self.x_data = x_data
        self.y_data = y_data

    def cost(self, params):
        func = self._model.copy()
        func.assign_params(params)
        res = self._cost_function(func(self.x_data),self.y_data)
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

    def train(self, cost, model, x_data, y_data=None, tolfun=1e-11):
        """The training algorithm"""
        super(CMA, self).train(cost, model, x_data, y_data)

        bmin, bmax = model.get_bounds()
        args = {'bounds': [bmin, bmax],
                'tolfun': tolfun, 'verb_log': 0}
        sigma = np.max(bmax)*0.1
        initsol = model.get_parameters()

        with closing(mp.Pool(self.num_cores)) as pool:
            es = CMAEvolutionStrategy(initsol, sigma, args)
            while not es.stop():
                solutions = es.ask()
                f_values = pool.map_async(self.cost, solutions).get()
                es.tell(solutions, f_values)
                es.logger.add()
                es.disp()
            print(es.result)
            pool.terminate()

        model.assign_params(es.result[0])
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

