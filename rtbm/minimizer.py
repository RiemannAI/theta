# -*- coding: utf-8 -*-

import time
from cma import CMAEvolutionStrategy
import multiprocessing as mp
from contextlib import closing
import numpy as np
from scipy.optimize import minimize


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
    res = resource.cost_function.cost(resource.model(resource.x_data), resource.y_data)
    return res


def worker_gradient(params):
    resource.model.set_parameters(params)
    xout = resource.model.feed_through(resource.x_data, True)
    resource.model.backprop(resource.cost_function.gradient(xout, resource.y_data))
    return resource.model.get_gradients()


class CMA(object):
    """Implements the GA using CMA library"""
    def __init__(self, parallel=False):
        super(CMA, self).__init__()
        if parallel:
            self.num_cores = mp.cpu_count()
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
                    solutions = es.ask(gradf=grad)
                    f_values = pool.map_async(worker_compute, solutions).get()
                    es.tell(solutions, f_values)
                    es.logger.add()
                    es.disp()
                pool.terminate()
        else:
            worker_initialize(cost, model, x_data, y_data)
            while not es.stop():
                solutions = es.ask(gradf=grad)
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


        
class SGD(object):
    """ Implements standard stochastic gradient descent """
    """ ToDo: Batch training """
    
    def train(self, cost, model, x_data, y_data=None, maxiter=100, lr=0.0001):

        t0 = time.time()
        for i in range(0, maxiter):
            Xout = model.feed_through(x_data, True)
            C = cost.cost(Xout,y_data)
            model.backprop(cost.gradient(Xout,y_data))
            
            # Get gradients
            G = model.get_gradients()
            W = model.get_parameters()
           
            # Adjust weights
            W = W - lr*G
            
            # Set gradients
            model.set_parameters(W)
            
            if(i % 100 == 0):
                print("Iteration %d in %.2f(s), cost = %f" % (i,time.time()-t0,C))
            
        print("Cost: ",C)    
        print("Sol: ",W)
        print("Time: %d s" % (time.time()-t0))
        return W


class BFGS(object):
    """Implements the BFGS method"""

    def train(self, cost, model, x_data, y_data=None, tolfun=1e-11, maxiter=100):
        """The training algorithm"""
        x0 = np.real(model.get_parameters())
        worker_initialize(cost, model, x_data, y_data)
        bounds = [ (model.get_bounds()[0][i],model.get_bounds()[1][i]) for i in range(model.size())]
        res = minimize(worker_compute, x0, jac=worker_gradient,
                       bounds=bounds, options = {'gtol': tolfun,
                                                 'disp': True, 'maxiter': maxiter})
        print(res)
        model.set_parameters(res.x)
        return res.x
