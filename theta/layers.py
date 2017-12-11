# -*- coding: utf-8 -*-
from __future__ import absolute_import

from abc import ABCMeta, abstractmethod
import numpy as np

from theta.mathtools import factorized_hidden_expectations,factorized_hidden_expectation_backprop
from theta.initializers import uniform, glorot_uniform, null
from theta.activations import tanh
from theta.riemann_theta.riemann_theta import RiemannTheta
from theta.rtbm import RTBM

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
        """
        Returns:
            int: total number of parameters.
        """
        return self._Np

    def set_bounds(self, param_bound):
        # Set bounds
        lower_bounds = [-param_bound for _ in range(self._Np)]
        upper_bounds = [ param_bound for _ in range(self._Np)]
        self._bounds = [lower_bounds, upper_bounds]
        
        
class NormAddLayer(Layer):
    r"""Linearly combines inputs with outputs normalized by sum of weights.
    Weights are exponentiated.

    .. math::
        M(v) = \frac{1}{\sum_{i=1}^N e^{\omega_i}} \sum_{i=1}^{N} e^{\omega_i} P^{(i)}(v)

    Args:
        Nin (int): number of input nodes.
        Nout (int): number of output nodes.
        W_init (theta.initializers): random initialization for weights.
        param_bound (float): maximum value alowed for the optimization via genetic optimizer.
    """

    def __init__(self, Nin, Nout, W_init=null(), param_bound=10):
        
        self._Nin = Nin
        self._Nout = Nout
        self._Np = self._Nout*self._Nin

        # Set paramter bounds
        self.set_bounds(param_bound)

        # Parameter init
        self._w = W_init.getinit((Nout,Nin))

    def feedin(self, X, grad_calc=False):
        """Feeds in the data X and returns the output of the layer.

        Args:
            X (numpy.array): the feedin data, shape (Nin, Ndata).
            grad_calc (bool): if True stores useful data for backpropagation.

        Returns:
            numpy.array: output of the layer.
            
        Note: 
            Results are vectorized.
        """
        eW = np.exp(self._w)
        S = np.sum(eW, axis=1)
        O = eW.dot(X)

        # Store data for grad calc
        if grad_calc:
            self._eW = eW
            self._X = X
            self._O = O
            self._S = S

        return O / S

    def backprop(self, E):
        """Evaluates and stores the gradients for backpropagation.

        Args:
            E (numpy.array): the error for backpropagation.

        Returns:
            numpy.array: the updated error function.
        """
        delta = self._X / self._S - self._O / self._S ** 2

        self._gradW = self._eW * E.dot(delta.T) / self._X.shape[1]

        return (self._eW / self._S).T.dot(E)

    def get_parameters(self):
        """
        Returns:
            numpy.array: the parameters as a flat array [w].
        """
        return self._w.flatten()

    def set_parameters(self, P):
        """Set the matrices from flat input array P
            P = [w]
        """

        self._w = P.reshape(self._w.shape)

        return True

    def get_gradients(self): 
        """
        Returns:
            numpy.array: W gradients as a flat array
        """
        return self._gradW.flatten()

    def get_bounds(self):
        """
        Returns:
            list of numpy.array: two arrays with min and max of each parameter of the layer for the GA.
        """
        return self._bounds


class Linear(Layer):
    """Linear layer.

    Args:
         Nin (int): number of inputs.
         Nout (int): number of outputs.
         W_init (theta.initializers): random initialization for weights.
         B_init (theta.initializers): random initialization for biases.
         param_bound (float): maximum value alowed for the optimization via genetic optimizer.
    """

    def __init__(self, Nin, Nout, W_init=glorot_uniform(), B_init=null(), param_bound=10):
        self._Nin  = Nin
        self._Nout = Nout
        self._Np = Nin*Nout+Nout

        self.set_bounds(param_bound)

        # Parameter init
        self._w = W_init.getinit((Nout,Nin) ).astype(float)
        self._b = B_init.getinit((Nout,1) ).astype(float)

    def feedin(self, X, grad_calc=False):
        """Feeds in the data X and returns the output of the layer.

        Args:
            X (numpy.array): the feedin data, shape (Nin, Ndata).
            grad_calc (bool): if True stores useful data for backpropagation.

        Returns:
            numpy.array: output of the layer.

        Note:
            Results are vectorized.
        """
        if(grad_calc==True):
            self._X = X

        return self._w.dot(X)+self._b

    def backprop(self, E):
        """Evaluates and stores the gradients for backpropagation.

        Args:
            E (numpy.array): the error for backpropagation.

        Returns:
            numpy.array: the updated error function.
        """

        # Mean bias gradient
        self._gradB = np.mean(E, axis=1,keepdims=True)

        # Mean weight gradient
        self._gradW = E.dot(self._X.T)/self._X.shape[1]

        # Propagate error
        return self._w.T.dot(E)

    def get_parameters(self):
        """
        Returns:
            numpy.array: the parameters as a flat array [b,w].
        """
        return np.concatenate((self._b.flatten(),self._w.flatten()))

    def set_parameters(self, params):
        """ Set the matrices from flat input array P
            P = [b,w]
        """
        index = 0

        self._b = params[index:index+self._b.shape[0]].reshape(self._b.shape)
        index += self._b.shape[0]

        self._w = params[index:index+self._w.size].reshape(self._w.shape)

        return True

    def get_gradients(self):
        """
        Returns:
            numpy.array: B and W gradients as a flat array
        """
        return np.concatenate((self._gradB.flatten(),self._gradW.flatten()))

    def get_bounds(self):
        """
        Returns:
            list of numpy.array: two arrays with min and max of each parameter of the layer for the GA.
        """
        return self._bounds


class NonLinear(Layer):
    """Non-Linear layer.

    Args:
        Nin (int): number of inputs.
        Nout (int): number of outputs.
        activation (theta.activations): the non-linear activation function.
        W_init (theta.initializers): random initialization for weights.
        B_init (theta.initializers): random initialization for biases.
        param_bound (float): maximum value alowed for the optimization via genetic optimizer.
    """

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

    def feedin(self, X, grad_calc=False):
        """Feeds in the data X and returns the output of the layer.

        Args:
            X (numpy.array): the feedin data, shape (Nin, Ndata).
            grad_calc (bool): if True stores useful data for backpropagation.

        Returns:
            numpy.array: output of the layer.

        Note:
            Results are vectorized.
        """
        # Calc linear map to activation ( X = previous outputs)
        L = self._w.dot(X) + self._b

        # Calc and store activation grad
        if (grad_calc == True):
            self._pO = X
            self._D = self._act.gradient(L)

        return self._act.activation(L)

    def backprop(self, E):
        """Evaluates and stores the gradients for backpropagation.

        Args:
            E (numpy.array): the error for backpropagation.

        Returns:
            numpy.array: the updated error function.
        """

        # Calc error at outputs
        Delta = np.multiply(self._D, E)

        # Mean bias gradient
        self._gradB = np.mean(Delta, axis=1, keepdims=True)

        # Mean weight gradient
        self._gradW = Delta.dot(self._pO.T) / self._pO.shape[1]

        # Propagate error
        return self._w.T.dot(Delta)
        
    def get_parameters(self):
        """
        Returns:
            numpy.array: the parameters as a flat array [b,w].
        """
        return np.concatenate((self._b.flatten(),self._w.flatten()))

    def set_parameters(self, params):
        """ Set the matrices from flat input array.

        Args:
            params (numpy.array): parameters [b,w]
        """
        index = 0

        self._b = params[index:index+self._b.shape[0]].reshape(self._b.shape)
        index += self._b.shape[0]

        self._w = params[index:index+self._w.size].reshape(self._w.shape)

        return True

    def get_gradients(self):
        """
        Returns:
            numpy.array: B and W gradients as a flat array
        """
        return np.concatenate((self._gradB.flatten(),self._gradW.flatten()))

    def get_bounds(self):
        """
        Returns:
            list of numpy.array: two arrays with min and max of each parameter of the layer for the GA.
        """
        return self._bounds


class DiagExpectationUnitLayer(Layer):
    """A layer of log-gradient theta units.

    Args:
        Nin (int): number of inputs.
        Nout (int): number of outputs.
        W_init (theta.initializers): random initialization for W
        B_init (theta.initializers): random initialization for B
        Q_init (theta.initializers): random initialization for Q
        param_bound (float): maximum value alowed for the optimization via genetic optimizer.
        phase (complex): the RTBM phase (default=1)
    """

    def __init__(self, Nin, Nout, W_init=glorot_uniform(),B_init=null(),Q_init=uniform(5,10+1e-5), param_bound=16, phase=1):
        self._Nin = Nin
        self._Nout = Nout
        self._phase = phase

        dtype = complex

        # Parameter init
        if phase == 1:
            dtype = float

        self._bh = phase*B_init.getinit((Nout,1)).astype(dtype)
        self._w = phase*W_init.getinit((Nin,Nout)).astype(dtype)

        self._q = np.diag(Q_init.getinit((Nout,))).astype(complex)

        self._Np = 2*self._Nout+self._Nout*self._Nin

        # Set bounds
        self.set_bounds(param_bound)

    def feedin(self, X, grad_calc=False):
        """Feeds in the data X and returns the output of the layer.

        Args:
            X (numpy.array): the feedin data, shape (Nin, Ndata).
            grad_calc (bool): if True stores useful data for backpropagation.

        Returns:
            numpy.array: output of the layer.

        Note:
            Results are vectorized.
        """
        vWb = np.transpose(X).dot(self._w) + self._bh.T

        if (grad_calc == True):
            self._X = X
            self._vWb = vWb

        if (self._phase == 1):
            return factorized_hidden_expectations(vWb, self._q, mode=1)
        else:
            return self._phase * factorized_hidden_expectations(vWb, self._q, mode=2)

    def backprop(self, E):
        """Evaluates and stores the gradients for backpropagation.

        Args:
            E (numpy.array): the error for backpropagation.

        Returns:
            numpy.array: the updated error function.
        """

        if self._phase == 1:
            Tn = factorized_hidden_expectation_backprop(self._vWb, self._q, mode=1)
        else:
            Tn = factorized_hidden_expectation_backprop(self._vWb, self._q, mode=2)

        kappa = -((Tn[1] - Tn[0] * Tn[0]) * 1.0 / (2j * np.pi) ** 2) * 1.0 / self._phase ** 2

        # B grad
        self._gradB = np.mean(kappa * E, axis=1, keepdims=True)

        # Q grad
        rho = + ((Tn[2] - Tn[0] * Tn[1]) * E * 1.0 / (2j * np.pi) ** 3) * 1.0 / self._phase ** 3

        self._gradQ = 0.5 * np.diag(np.mean(rho, axis=1).flatten())

        # W grad
        delta = kappa * E

        self._gradW = delta.dot(self._X.T).T / self._X.shape[1]

        return 1.0 / self._phase * self._w.dot(delta)

    def set_bounds(self, param_bound):
        """Set bounds for GA.

        Args:
            param_bound (float): min and max of each parameter when using GA.
        """
        lower_bounds = [-param_bound for _ in range(self._Np)]
        upper_bounds = [ param_bound for _ in range(self._Np)]
        self._bounds = [lower_bounds, upper_bounds]

        # set special q bounds
        index = self._Np-self._q.shape[0]
        lower_bounds[index:] = [1e-5]*self._q.shape[0]
        upper_bounds[index:] = [param_bound]*self._q.shape[0]
        
    def show_activation(self, N, bound=2):
        """Plots the Nth activation function on [-bound,+bound].

        Args:
            N (int): the Nth activation function
            bound (float): min/max value for the plot.
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
        """
        Returns:
            numpy.array: the parameters as a flat array [bh,w,q]
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
        """
        Returns:
            list of numpy.array: two arrays with min and max of each parameter of the layer for the GA.
        """
        return self._bounds

    def get_gradients(self):
        """
        Returns:
            numpy.array: B, W and Q gradients as a flat array
        """
        return np.concatenate((self._gradB.flatten(),self._gradW.flatten(),self._gradQ.diagonal()))


class ThetaUnitLayer(Layer):
    """Allocate a Theta Unit Layer working in probability mode

    Args:
        Nin (int): number of input nodes
        Nout (int): number of output nodes (i.e. # of RTBMs)
        Nhidden (int): number of hidden layers per RTBM
        init_max_param_bound (float): maximum bound value for CMA
        random_bound (float): the maximum value for the random matrix X used by initialization
        phase (complex): number which multiplies w and bh ``phase=1`` for Phase I and ``phase=1j`` for PhaseII.
        diagonal_T (bool): force T diagonal, by default T is symmetric.
    """

    def __init__(self, Nin, Nout, Nhidden=1, init_max_param_bound=2, random_bound=1, phase=1, diagonal_T=False):
        self._Nin = Nin
        self._Nout = Nout

        self._rtbm = []
        for m in range(Nout):
            self._rtbm.append(RTBM(Nin, Nhidden, init_max_param_bound=init_max_param_bound,
                                   random_bound=random_bound, phase=phase, diagonal_T=diagonal_T))

        self._Np = np.sum([r.size() for r in self._rtbm])

        self._bounds = None
        self.set_bounds()

    def get_unit(self, N):
        """Return the singular RTBM unit.

        Args:
            N (int): the Nth RTBM unit.

        Returns:
            theta.rtbm.RTBM: the Nth RTBM unit.
        """
        if 0 < N <= len(self._rtbm):
            return self._rtbm[N-1]
        else:
            print("Layer consists of only",len(self._rtbm),"units")
            return None
        
    def feedin(self, X, grad_calc=False):
        """Feeds in the data X and returns the output of the layer.

        Args:
            X (numpy.array): the feedin data, shape (Nin, Ndata).
            grad_calc (bool): if True stores useful data for backpropagation.

        Returns:
            numpy.array: output of the layer.

        Note:
            Results are vectorized.
        """
        result = np.zeros(shape=(self._Nout, X.shape[1]), dtype=float)
        for i, m in enumerate(self._rtbm):
            result[i] = m(X, grad_calc=grad_calc)
        return result

    def backprop(self, E):
        """Evaluates and stores the gradients for backpropagation.

        Args:
            E (numpy.array): the error for backpropagation.

        Returns:
            numpy.array: the updated error function.
        """
        result = np.zeros(shape=(self._Nout, E.shape[1]), dtype=float)

        for i, m in enumerate(self._rtbm):
            result[i] = m.backprop(E[i, :])

        """ 
        Currently only as one layer supported 
        Flows from individual RTBMs need to be aggregated before 
        moved back further into shared inputs
        """

    def get_parameters(self):
        """
        Returns:
            numpy.array: the parameters as a flat array [b,w,q].
        """
        params = np.zeros(shape=(self._Np))

        index = 0
        for m in self._rtbm:
            params[index:index+m.size()] = m.get_parameters()
            index += m.size()

        return params

    def set_parameters(self, params):
        """Set the matrices from flat input array.

        Args:
            params (numpy.array): parameters for the RTBM units.

        Returns:
            bool: True if the positivity condition is satisfied, False otherwise.
        """
        index = 0
        for m in self._rtbm:
            if not m.set_parameters(params[index:index+m.size()]):
                return False
            index += m.size()
        return True

    def set_bounds(self, *params):
        """Set bounds for GA.

        Args:
            param_bound (float): min and max of each parameter when using GA.
        """
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
        """
        Returns:
            list of numpy.array: two arrays with min and max of each parameter of the layer for the GA.
        """
        return self._bounds

    def get_gradients(self):
        """
        Returns:
            numpy.array: gradients for all RTBM units as a flat array.
        """
        grads = np.zeros(shape=(self._Np))

        index = 0
        for m in self._rtbm:
            grads[index:index+m.size()] = m.get_gradients()
            index += m.size()
        
        return grads