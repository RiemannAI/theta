# -*- coding: utf-8 -*-

from cma import CMAEvolutionStrategy
import multiprocessing as mp
from contextlib import closing
import numpy as np


class Resource(object):
    """Resources container for the worker load"""
    def __init__(self, cost, model, x_data, y_data=None):
        self.cost_function = cost
        self.model = model
        self.x_data = x_data
        self.y_data = y_data


def worker_initialize(cost, model, x_data, y_data):
    global resource
    resource = Resource(cost, model, x_data, y_data)


def worker_compute(params):
    resource.model.set_parameters(params)
    res = resource.cost_function(resource.model(resource.x_data), resource.y_data)
    return res


class CMA(object):
    """Implements the GA using CMA library"""
    def __init__(self, parallel=False):
        super(CMA, self).__init__()
        if parallel:
            self.num_cores = mp.cpu_count()
        else:
            self.num_cores = 1
        print('CMA on %d cpu(s) enabled' % self.num_cores)

    def train(self, cost, model, x_data, y_data=None, tolfun=1e-11, popsize=None, maxiter=None):
        """The training algorithm"""

        initsol = model.get_parameters()
        args = {'bounds': model.get_bounds(),
                'tolfun': tolfun,
                'verb_log': 0}
        sigma = np.max(model.get_bounds()[1])*0.1

        if popsize is not None:
            args['popsize'] = popsize

        if maxiter is not None:
            args['maxiter'] = maxiter
          
        es = CMAEvolutionStrategy(initsol, sigma, args)
        if self.num_cores > 1:
            with closing(mp.Pool(self.num_cores, initializer=worker_initialize,
                                 initargs=(cost, model, x_data, y_data))) as pool:
                while not es.stop():
                    solutions = es.ask()
                    f_values = pool.map_async(worker_compute, solutions).get()
                    es.tell(solutions, f_values)
                    es.logger.add()
                    es.disp()
                pool.terminate()
        else:
            worker_initialize(cost, model, x_data, y_data)
            while not es.stop():
                solutions = es.ask()
                f_values = [ worker_compute(isol) for isol in solutions ]
                es.tell(solutions, f_values)
                es.logger.add()
                es.disp()
        print(es.result)

        model.set_parameters(es.result[0])
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

