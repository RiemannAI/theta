# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import

import numpy as np
import multiprocessing as mp

from cma import CMAEvolutionStrategy
from contextlib import closing
from scipy.optimize import minimize

import theta.sgd as sgd


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
    """Implements the GA using CMA-ES library (cma package).
    This class provides a basic CMA-ES implementation for RTBMs.

    Args:
        parallel (bool): if set to True the algorithm uses multi-processing.
        ncores (int): limit the number of cores when ``parallel=True``.
    """
    def __init__(self, parallel=False, ncores=0, verbose=True):
        self._verbose = verbose
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
        """Trains the ``model`` using the custom `cost` function.

        Args:
            cost (theta.costfunctions): the cost function.
            model (theta.model.Model or theta.rtbm.RTBM): the model to be trained.
            x_data (numpy.array): the support data with shape (Nv, Ndata).
            y_data (numpy.array): the target prediction.
            tolfun (float): the maximum tolerance of the cost function fluctuation to stop the minimization.
            popsize (int): the population size.
            maxiter (int): the maximum number of iterations.
            use_grad (bool): if True the gradients for the cost and model are used in the minimization.

        Returns:
            numpy.array: the optimal parameters

        Note:
            The parameters of the model are changed by this algorithm.
        """
        if self._verbose:
            vb = 0
        else:
            vb = -9

        initsol = np.real(model.get_parameters())

        # Prepare the bounds
        bounds = model.get_bounds()
        # The diagonal must be always positive
        t_diagonal = np.diag(model._t)
        q_diagonal = np.diag(model._q)
        a_diagonal = np.concatenate([t_diagonal, q_diagonal])
        for val in a_diagonal:
            idx = np.where(initsol == val)[0][0]
            bounds[0][idx] = np.maximum(1e-7, bounds[0][idx])

        args = {'bounds': bounds,
                'tolfun': tolfun,
                'verbose': vb,
                'verb_log': vb}
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
    """Stochastic gradient descent."""

    def train(self, cost, model, x_data, y_data=None,
              validation_split=0, validation_x_data=None, validation_y_data=None, stopping=None,
              scheme=None, maxiter=100, batch_size=0, shuffle=False, lr=0.001, decay=0, momentum=0,nesterov=False, noise=0,cplot=True):
        """Trains the given model with stochastic gradient descent methods

        Args:
            cost (theta.costfunctions): the cost fuction class
            model (theta.rtbm.Model or theta.model.Model): the model to be trained
            x_data (numpy.array): the target data support
            y_data (numpy.array): the target data prediction
            validation_split (float): fraction of data used for validation only
            validation_x_data (numpy.array): external set of validation support
            validation_y_data (numpy.array): external set of validation target
            stopping (theta.stopping): the stopping class (see ``theta.stopping``)
            scheme (theta.gradientscheme): the SGD method (Ada, RMSprop, see `Gradient descent schemes`_)
            maxiter (int): maximum number of allowed iterations
            batch_size (int): the batch size
            shuffle (bool): shuffle the data on each iteration
            lr (float): learning rate
            decay (float): learning rate decay rate
            momentum (float): add momentum
            nesterov (bool): add nesterov momentum
            noise (bool): add gaussian noise
            cplot (bool): if True shows the cost function evolution

        Returns:
            dictionary: iterations, cost and validation functions

        Note:
            The parameters of the model are changed by this algorithm.
        """
        return sgd.train(cost, model, x_data, y_data, validation_split, validation_x_data, validation_y_data, stopping,
                         scheme, maxiter, batch_size,shuffle, lr, decay, momentum, nesterov, noise, cplot)
    

class BFGS(object):
    """Implements the BFGS method"""

    def train(self, cost, model, x_data, y_data=None, tolfun=1e-11, maxiter=100):
        """
        Args:
            cost (theta.costfunctions): the cost function.
            model (theta.model.Model or theta.rtbm.RTBM): the model to be trained.
            x_data (numpy.array): the support data with shape (Nv, Ndata).
            y_data (numpy.array): the target prediction.
            tolfun (float): the maximum tolerance of the cost function fluctuation to stop the minimization.
            popsize (int): the population size.
            maxiter (int): the maximum number of iterations.

        Returns:
            numpy.array: the optimal parameters

        Note:
            The parameters of the model are changed by this algorithm.
        """
        x0 = np.real(model.get_parameters())
        worker_initialize(cost, model, x_data, y_data)
        bounds = [ (model.get_bounds()[0][i],model.get_bounds()[1][i]) for i in range(model.size())]
        res = minimize(worker_compute, x0, jac=lambda x: np.ascontiguousarray(worker_gradient(x), dtype=np.double),
                       bounds=bounds, options = {'gtol': tolfun,
                                                 'disp': True, 'maxiter': maxiter})
        print(res)
        model.set_parameters(res.x)
        return res.x
