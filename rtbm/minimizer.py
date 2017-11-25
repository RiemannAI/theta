# -*- coding: utf-8 -*-

from __future__ import print_function
from cma import CMAEvolutionStrategy
import multiprocessing as mp
from contextlib import closing
import numpy as np
from scipy.optimize import minimize
import sgd


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
    if not resource.model.set_parameters(params):
        return np.NaN
    res = resource.cost_function.cost(np.real(resource.model(resource.x_data)), resource.y_data)
    return res


def worker_gradient(params):
    resource.model.set_parameters(params)
    xout = resource.model.feed_through(resource.x_data, True)
    resource.model.backprop(resource.cost_function.gradient(xout, resource.y_data))
    return resource.model.get_gradients()


class CMA(object):
    """Implements the GA using CMA library"""
    def __init__(self, parallel=False, ncores=0):
        super(CMA, self).__init__()
        if parallel:
            if(ncores==0):
                self.num_cores = mp.cpu_count()
            else:
                self.num_cores = min(ncores,mp.cpu_count())
        else:
            self.num_cores = 1
        print('CMA on %d cpu(s) enabled' % self.num_cores)

    def train(self, cost, model, x_data, y_data=None, tolfun=1e-11, popsize=None, maxiter=None, use_grad=False):
        """The training algorithm"""

        initsol = np.real(model.get_parameters())
        args = {'bounds': model.get_bounds(),
                'tolfun': tolfun,
                'verb_log': 0}
        sigma = np.max(model.get_bounds()[1])*0.1

        if popsize is not None:
            args['popsize'] = popsize

        if maxiter is not None:
            args['maxiter'] = maxiter

        grad = None
        if use_grad:
            grad = worker_gradient

        es = CMAEvolutionStrategy(initsol, sigma, args)
        if self.num_cores > 1:
            with closing(mp.Pool(self.num_cores, initializer=worker_initialize,
                                 initargs=(cost, model, x_data, y_data))) as pool:
                while not es.stop():
                    f_values, solutions = [], []
                    while len(solutions) < es.popsize:
                        x = es.ask(es.popsize-len(solutions), gradf=grad)
                        curr_fit = pool.map_async(worker_compute, x).get()
                        for value, solution in zip(curr_fit,x):
                            if not np.isnan(value):
                                solutions.append(solution)
                                f_values.append(value)
                    es.tell(solutions, f_values)
                    es.disp()
                pool.terminate()
        else:
            worker_initialize(cost, model, x_data, y_data)
            while not es.stop():
                f_values, solutions = [], []
                while len(solutions) < es.popsize:
                    curr_fit = x = np.NaN
                    while np.isnan(curr_fit):
                        x = es.ask(1, gradf=grad)[0]
                        curr_fit = worker_compute(x)
                    solutions.append(x)
                    f_values.append(curr_fit)
                es.tell(solutions, f_values)
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


class SGD(object):
    """Stochastic gradient descent"""

    def train(self, cost, model, x_data, y_data=None, validation_split=0, scheme=None, maxiter=100, batch_size=0,shuffle=False,
              lr=0.001, decay=0, momentum=0,nesterov=False, noise=0,cplot=True):
        """Trains the given model with stochastic gradient descent methods

        :param cost: the cost fuction class
        :param model: the model to be trained
        :param x_data: the target data support
        :param y_data: the target data prediction
        :param scheme: the SGD method (Ada, RMSprop, see gradientschemes.py)
        :param maxiter: maximum number of allowed iterations
        :param batch_size: the batch size
        :param shuffle: shuffle the data on each iteration
        :param lr: learning rate
        :param decay: learning rate decay rate
        :param momentum: add momentum
        :param nesterov: add nesterov momentum
        :param noise: add gaussian noise
        :param cplot: if True shows the cost function evolution
        :return: dictionary with iterations and cost functions
        """
        
        return sgd.train(cost, model, x_data, y_data, validation_split, scheme, maxiter, batch_size,shuffle,
                         lr, decay, momentum, nesterov, noise, cplot)
    

class BFGS(object):
    """Implements the BFGS method"""

    def train(self, cost, model, x_data, y_data=None, tolfun=1e-11, maxiter=100):
        """The training algorithm"""
        x0 = np.real(model.get_parameters())
        worker_initialize(cost, model, x_data, y_data)
        bounds = [ (model.get_bounds()[0][i],model.get_bounds()[1][i]) for i in range(model.size())]
        res = minimize(worker_compute, x0, jac=lambda x: np.ascontiguousarray(worker_gradient(x), dtype=np.double),
                       bounds=bounds, options = {'gtol': tolfun,
                                                 'disp': True, 'maxiter': maxiter})
        print(res)
        model.set_parameters(res.x)
        return res.x
