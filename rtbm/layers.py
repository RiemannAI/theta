# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np
from mathtools import factorized_hidden_expectations,factorized_hidden_expectation_backprop
from riemann_theta.riemann_theta import RiemannTheta
from initializers import uniform, normal, glorot_uniform, glorot_normal, null
from activations import sigmoid, tanh
from rtbm import RTBM

import matplotlib.pyplot as plt


class Layer(object):
    """ Abstract class for a layer of a deep network """
    __metaclass__ = ABCMeta

    @abstractmethod
    def feedin(self, X, *grad_calc):
        pass

    @abstractmethod
    def get_parameters(self):
        pass

    @abstractmethod
    def set_parameters(self, params):
        pass

    @abstractmethod
    def size(self):
        pass

    @abstractmethod
    def get_bounds(self):
        pass

    def get_Nin(self):
        return self._Nin

    def get_Nout(self):
        return self._Nout

    def size(self):
        """ Returns total # parameters """
        return self._Np

    def set_bounds(self, param_bound):
        # Set bounds
        lower_bounds = [-param_bound for _ in range(self._Np)]
        upper_bounds = [ param_bound for _ in range(self._Np)]
        self._bounds = [lower_bounds, upper_bounds]
        
        
class NormAddLayer(Layer):
    """ Linearly combines inputs with outputs normalized by sum of weights """
    """ (no bias) """

    def __init__(self, Nin, Nout, param_bound=10):
        self._Nin = Nin
        self._Nout = Nout
        self._Np = self._Nout*self._Nin

        # Set paramter bounds
        self.set_bounds(param_bound)

        # Parameter init
        self._w = np.random.uniform(-param_bound, param_bound,(Nout,Nin)).astype(complex)


    def get_parameters(self):
        """ Returns the parameters as a flat array
            [w]
        """

        return self._w.flatten()

    def set_parameters(self, P):
        """ Set the matrices from flat input array P
            P = [w]
        """

        self._w = P.reshape(self._w.shape)

        return True

    def get_bounds(self):
        """Returns two arrays with min and max of each parameter for the GA"""

        return self._bounds


    def feedin(self, X, *grad_calc):
        """ Feeds in the data X and returns the output of the layer
            Note: Vectorized
        """

        S = np.sum(self._w,axis=1)
        O = self._w.dot(X)

        return np.divide(O, S[:, np.newaxis])



class Linear(Layer):
    """ Linear layer """
    def __init__(self, Nin, Nout, W_init=glorot_uniform(), B_init=null(), param_bound=10):
        self._Nin  = Nin
        self._Nout = Nout
        self._Np = Nin*Nout+Nout

        self.set_bounds(param_bound)

        # Parameter init
        self._w = W_init.getinit((Nout,Nin) ).astype(float)
        self._b = B_init.getinit((Nout,1) ).astype(float)

    

    def get_parameters(self):
        """ Returns the parameters as a flat array
            [b,w]
        """

        return np.concatenate((self._b.flatten(),self._w.flatten()))

    def get_gradients(self):
        """ Returns gradients as a flat array
            [b,w]
        """
        return np.concatenate((self._gradB.flatten(),self._gradW.flatten()))


    def feedin(self, X, grad_calc=False):
        """ Feeds in the data X and returns the output of the layer
            Note: Vectorized
        """
        if(grad_calc==True):
            self._X = X

        return self._w.dot(X)+self._b

    def backprop(self, E):
        """ Propagates the error E through the layer and stores gradient """

        # Mean bias gradient
        self._gradB = np.mean(E, axis=1,keepdims=True)

        # Mean weight gradient
        self._gradW = E.dot(self._X.T)/self._X.shape[1]

        # Propagate error
        return self._w.T.dot(E)


    def set_parameters(self, params):
        """ Set the matrices from flat input array P
            P = [b,w]
        """
        index = 0

        self._b = params[index:index+self._b.shape[0]].reshape(self._b.shape)
        index += self._b.shape[0]

        self._w = params[index:index+self._w.size].reshape(self._w.shape)

        return True


    def get_bounds(self):
        """Returns two arrays with min and max of each parameter for the GA"""
        return self._bounds



class NonLinear(Layer):
    """ Non-Linear layer """

    def __init__(self, Nin, Nout, activation=tanh, W_init=glorot_uniform(), B_init=null(), param_bound=10):
        self._Nin  = Nin
        self._Nout = Nout
        self._Np = Nin*Nout+Nout
        self._act = activation

        # Set bounds
        self.set_bounds(param_bound)
        
        # Parameter init
        self._w = W_init.getinit((Nout,Nin)).astype(float)
        self._b = B_init.getinit((Nout,1)).astype(float)

        
    def get_parameters(self):
        """ Returns the parameters as a flat array
            [b,w]
        """

        return np.concatenate((self._b.flatten(),self._w.flatten()))

    def get_gradients(self):
        """ Returns gradients as a flat array
            [b,w]
        """
        return np.concatenate((self._gradB.flatten(),self._gradW.flatten()))


    def feedin(self, X, grad_calc=False):
        """ Feeds in the data X and returns the output of the layer
            Note: Vectorized
        """
        # Calc linear map to activation ( X = previous outputs)
        L = self._w.dot(X)+self._b;
        
        # Calc and store activation grad
        if(grad_calc==True):
            self._pO = X
            self._D = self._act.gradient(L)

        return self._act.activation(L)

    def backprop(self, E):
        """ Propagates the error E through the layer and stores gradient """

        # Calc error at outputs
        Delta = np.multiply(self._D,E)

        # Mean bias gradient
        self._gradB = np.mean(Delta, axis=1,keepdims=True)

        # Mean weight gradient
        self._gradW = Delta.dot(self._pO.T)/self._pO.shape[1]

        # Propagate error
        return self._w.T.dot(Delta)


    def set_parameters(self, params):
        """ Set the matrices from flat input array P
            P = [b,w]
        """
        index = 0

        self._b = params[index:index+self._b.shape[0]].reshape(self._b.shape)
        index += self._b.shape[0]

        self._w = params[index:index+self._w.size].reshape(self._w.shape)

        return True


    def get_bounds(self):
        """Returns two arrays with min and max of each parameter for the GA"""
        return self._bounds



class SoftMaxLayer(Layer):
    """ A layer to perform the softmax operation """
    def __init__(self, Nin):
        self._Nin  = Nin
        self._Nout = Nin
        self._Np = 0
        self._param_bound = 0

    def get_parameters(self):
        """ Returns the parameters as a flat array
            []
        """
        return np.empty(0)

    def get_gradients(self):
        """ Returns gradients as a flat array
            []
        """
        return np.empty(0)

    def set_parameters(self, params):
        return True


    def get_bounds(self):
        """Returns two arrays with min and max of each parameter for the GA"""
        self._lower_bounds = []
        self._upper_bounds = []

        return [self._lower_bounds, self._upper_bounds]

    def feedin(self, X, grad_calc=False):
        """ Feeds in the data X and returns the output of the layer
            Note: Vectorized
        """
        E = np.exp(X)
        S = np.sum(E,axis=0)

        O = np.divide(E, S[np.newaxis,:])

        # Store O for backprop
        if(grad_calc==True):
            self._pO = O

        return O

    def backprop(self, E):
        """ Propagates the error E through the layer """

        # Propagate error
        return E*self._pO+self._pO.dot(E.T.dot(self._pO))


class MaxPosLayer(Layer):
    """ Depreciated

    """

    def __init__(self, Nin, startPos=0):
        self._Nin = Nin
        self._Nout = 1
        self._Np = 0
        self._param_bound = 0
        self._startPos = startPos

    def get_parameters(self):
        """ Returns the parameters as a flat array
            []
        """

        return np.empty(0)

    def set_parameters(self, params):
        return True


    def get_bounds(self):
        """Returns two arrays with min and max of each parameter for the GA"""
        self._lower_bounds = []
        self._upper_bounds = []

        return [self._lower_bounds, self._upper_bounds]

    def feedin(self, X, *grad_calc):
        """ Feeds in the data X and returns the output of the layer
            Note: Vectorized
        """

        return np.argmax(X,axis=0)+self._startPos



class DiagExpectationUnitLayer(Layer):
    """ A layer of log-gradient theta units """

    def __init__(self, Nin, Nout, W_init=glorot_uniform(),B_init=null(),Q_init=uniform(5,10+1e-5), param_bound=16, phase=1):
        self._Nin = Nin
        self._Nout = Nout
        self._phase = phase

        dtype = complex

        # Parameter init
        if(phase == 1):
            dtype = float

        self._bh = phase*B_init.getinit((Nout,1)).astype(dtype)
        self._w = phase*W_init.getinit((Nin,Nout)).astype(dtype)

        self._q = np.diag(Q_init.getinit((Nout,))).astype(complex)

        self._Np = 2*self._Nout+self._Nout*self._Nin

        # Set bounds
        self.set_bounds(param_bound)
        
        """ ToDo: bound check for init """

    def set_bounds(self, param_bound):   
        # Set bounds
        lower_bounds = [-param_bound for _ in range(self._Np)]
        upper_bounds = [ param_bound for _ in range(self._Np)]
        self._bounds = [lower_bounds, upper_bounds]

        # set special q bounds
        index = self._Np-self._q.shape[0]
        lower_bounds[index:] = [1e-5]*self._q.shape[0]
        upper_bounds[index:] = [param_bound]*self._q.shape[0]
        
    def show_activation(self, N, bound=2):
        """
            Plots the Nth activation function on
            [-bound,+bound]
        """
        if(N > self._Nout):
            print("Node does not exist!")
        else:

            D = self._phase*np.linspace(-bound, bound, 1000)
            D = D.reshape((D.shape[0],1))

            O = np.matrix([[self._q[N-1, N-1]]], dtype=complex)

            if(self._phase==1):
                E = -1.0/(2j*np.pi)*RiemannTheta.normalized_eval(D / (2.0j * np.pi), -O/ (2.0j * np.pi), mode=1, derivs=[[1]])
            else:
                E = -1.0/(2j*np.pi)*self._phase*RiemannTheta.normalized_eval(D / (2.0j * np.pi), -O/ (2.0j * np.pi), mode=2, derivs=[[1]])

            plt.plot(1.0/self._phase*D.flatten(), E.flatten(),"b-")
            
    def get_parameters(self):
        """ Returns the parameters as a flat array
            [bh,w,q]
        """

        return np.concatenate((self._phase*self._bh.flatten(),1.0/self._phase*self._w.flatten(),self._q.diagonal()))

    def set_parameters(self, params):
        """ Set the matrices from flat input array P
            P = [bh,w,q]
        """
        index = 0

        self._bh = self._phase*params[index:index+self._bh.shape[0]].reshape(self._bh.shape)
        index += self._bh.shape[0]

        self._w = self._phase*params[index:index+self._w.size].reshape(self._w.shape)
        index += self._w.size

        np.fill_diagonal(self._q, params[index:index+self._q.shape[0]])

        return True

    def get_bounds(self):
        """Returns two arrays with min and max of each parameter for the GA"""
        return self._bounds

    def get_gradients(self):
        """ Returns gradients as a flat array
            [b,w,q]
        """
        return np.concatenate((self._gradB.flatten(),self._gradW.flatten(),self._gradQ.diagonal()))

    def feedin(self, X, grad_calc=False):
        """ Feeds in the data X and returns the output of the layer
            Note: Vectorized
        """
        vWb = np.transpose(X).dot(self._w)+self._bh.T

        if(grad_calc==True):
            self._X = X
            self._vWb = vWb

        if(self._phase==1):
            return factorized_hidden_expectations(vWb, self._q, mode=1)
        else:
            return self._phase*factorized_hidden_expectations(vWb, self._q, mode=2)

    
    def backprop(self, E):
        """ Propagates the error E through the layer and stores gradient """
        
        if(self._phase==1):
            Tn = factorized_hidden_expectation_backprop(self._vWb, self._q, mode=1)
        else:
            Tn = factorized_hidden_expectation_backprop(self._vWb, self._q, mode=2)

        kappa = -( ( Tn[1] - Tn[0]*Tn[0] )*1.0/(2j*np.pi)**2 )*1.0/self._phase**2

        # B grad
        self._gradB = np.mean(kappa*E,axis=1,keepdims=True)

        # Q grad
        rho = + ( (Tn[2] - Tn[0]*Tn[1] )*E*1.0/(2j*np.pi)**3 )*1.0/self._phase**3

        self._gradQ = 0.5 * np.diag(np.mean(rho, axis=1).flatten())

        # W grad
        delta = kappa*E

        self._gradW = delta.dot(self._X.T).T/self._X.shape[1]
        
        return 1.0/self._phase*self._w.dot(delta)


class ThetaUnitLayer(Layer):
    """ A layer of theta units """

    def __init__(self, Nin, Nout, Nhidden=1, init_max_param_bound=2, random_bound=1, phase=1, diagonal_T=False):
        """Allocate a Theta Unit Layer working in probability mode

        :param Nin: number of input nodes
        :param Nout: number of output nodes (i.e. # of RTBMs)
        :param Nhidden: number of hidden layers per RTBM
        :param init_max_param_bound: maximum bound value for CMA
        :param random_bound: the maximum value for the random matrix X used by the Schur complement
        :param phase: number which multiplies w and bh
        :param diagonal_T: force T diagonal, by default T is symmetric
        """

        self._Nin = Nin
        self._Nout = Nout

        self._rtbm = []
        for m in range(Nout):
            self._rtbm.append(RTBM(Nin, Nhidden, init_max_param_bound=init_max_param_bound,
                                   random_bound=random_bound, phase=phase, diagonal_T=diagonal_T))

        self._Np = np.sum([r.size() for r in self._rtbm])

        self._bounds = None
        self.set_bounds()

    def feedin(self, X, grad_calc=False):
        """ Feeds in the data X and returns the output of the layer
            Note: Vectorized
        """
        result = np.zeros(shape=(self._Nout, X.shape[1]), dtype=float)
        for i, m in enumerate(self._rtbm):
            result[i] = m(X, grad_calc=grad_calc)
        return result

    def get_parameters(self):
        """ Returns the parameters as a flat array
            [bh,w,q]
        """
        params = np.zeros(shape=(self._Np))

        index = 0
        for m in self._rtbm:
            params[index:index+m.size()] = m.get_parameters()
            index += m.size()

        return params

    def set_parameters(self, params):
        """ Set parameters"""
        index = 0
        for m in self._rtbm:
            if not m.set_parameters(params[index:index+m.size()]):
                return False
            index += 0
        return True

    def set_bounds(self, *params):
        """Compute bounds"""
        lower_bounds = np.zeros(shape=(self._Np))
        upper_bounds = np.zeros(shape=(self._Np))
        index = 0
        for m in self._rtbm:
            bound = m.get_bounds()
            lower_bounds[index:index+m.size()] = bound[0]
            upper_bounds[index:index+m.size()] = bound[1]
            index += m.size()
        self._bounds = [lower_bounds.tolist(), upper_bounds.tolist()]

    def get_bounds(self):
        """Returns two arrays with min and max of each parameter for the GA"""
        return self._bounds

    def get_gradients(self):
        """ Returns gradients as a flat array
        """
        grads = np.zeros(shape=(self._Np))

        index = 0
        for m in self._rtbm:
            grads[index:index+m.size()] = m.get_gradients()
            index += m.size()
        
        return grads
            
    def backprop(self, E):
        """ Propagates the error E through the layer and stores gradient """
        
        result = np.zeros(shape=(self._Nout, E.shape[1]), dtype=float)
        
        for i, m in enumerate(self._rtbm):
            result[i] = m.backprop(E)
            
        """ Currently only as one layer supported 
            Flows from individual RTBMs need to be aggregated before 
            moved back further into shared inputs
        """
        pass
        