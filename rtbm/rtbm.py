# -*- coding: utf-8 -*-

import numpy as np
from mathtools import rtbm_probability, hidden_expectations, rtbm_log_probability, \
    check_normalization_consistency, check_pos_def


class AssignError(Exception):
    pass


class RTBM(object):
    """This class implements the Riemann Theta Boltzmann Machine"""
    class Mode:
        Probability = 0
        LogProbability = 1
        Expectation = 2

    def __init__(self, visible_units, hidden_units, mode=Mode.Probability,
                 init_max_param_bound=2, random_bound=1, phase=1):
        """Setup operators for BM based on the number of visible and hidden units

        Args:
            visible_units: number of visible units
            hidden_units: number of hidden units
            mode: see Mode enumerator
            init_max_param_bound: size of maximum parameters used in random initialization.
            random_bound: selects the maximum random value for the Schur complement initialization
            phase: number which multiplies w and bh

        """
        self._Nv = visible_units
        self._Nh = hidden_units
        self._bv = np.zeros([visible_units, 1])
        self._t = np.ones([visible_units, visible_units])
        self._bh = np.zeros([hidden_units, 1])
        self._w = np.zeros([visible_units, hidden_units])
        self._q = np.zeros([hidden_units, hidden_units])
        self._size = self._Nv + self._Nh + (self._Nv**2+self._Nv+self._Nh**2+self._Nh)/2+self._Nv*self._Nh
        self._mode = None
        self._call = None
        self._parameters = None
        self._phase = phase

        # set operation mode
        self.mode = mode

        # set boundaries
        self.set_bounds(init_max_param_bound)
        
        # Populate with random parameters using Schur complement
        # This guarantees an acceptable and instantaneous initial solution
        self.random_init(random_bound)

    def __call__(self, data):
        """Evaluates the RTBM instance for a given data array"""
        return self._call(data)

    def size(self):
        """Get size of RTBM"""
        return self._size

    def random_init(self, bound):
        """A fast random initializer based on Schur complement

        Args:
            bound: the maximum value for the random matrix X used by the Schur complement
        """

        a_shape = ((self._Nv+self._Nh), (self._Nv+self._Nh))
        a_size = (self._Nv+self._Nh)**2

        params = np.random.uniform(-bound, bound, a_size+self._Nv+self._Nh)
        x = params[:a_size].reshape(a_shape)
        a = np.transpose(x).dot(x)

        self._q = a[:self._Nh, :self._Nh]
        self._t = a[self._Nh:self._Nh + self._Nv, self._Nh:]
        self._w = self._phase * a[self._Nh:, :self._Nh]
        self._bv = params[a_size:a_size + self._Nv].reshape(self._bv.shape)
        self._bh = self._phase * params[-self._Nh:].reshape(self._bh.shape)

        # store parameters having in mind that Q and T are symmetric.
        self._parameters = np.concatenate([self._bv.flatten(), self._bh.flatten(),
                                           self._w.flatten(), self._t[np.triu_indices(self._Nv)],
                                           self._q[np.triu_indices(self._Nh)]])

    def set_parameters(self, params):
        """Assigns a flat array of parameters to the RTBM matrices.

        Args
            params: list of parameters to populate Bv, Bh, W, T, Q

        Returns:
            True if Q, T and Q-WtTW are positive, False otherwise.
        """

        if len(params) != self._size:
            raise Exception('Size does no match.')

        self._parameters = params

        self._bv = params[:self._Nv].reshape(self._bv.shape)
        index = self._Nv

        self._bh = self._phase*params[index:index+self._Nh].reshape(self._bh.shape)
        index += self._Nh

        self._w = self._phase*params[index:index+self._Nv*self._Nh].reshape(self._w.shape)
        index += self._w.size

        inds = np.triu_indices_from(self._t)
        self._t[inds] = params[index:index+(self._Nv**2+self._Nv)/2]
        self._t[(inds[1], inds[0])] = params[index:index+(self._Nv**2+self._Nv)/2]
        index += (self._Nv**2+self._Nv)/2

        inds = np.triu_indices_from(self._q)
        self._q[inds] = params[index:index+(self._Nh**2+self._Nh)/2]
        self._q[(inds[1], inds[0])] = params[index:index+(self._Nh**2+self._Nh)/2]

        if not check_normalization_consistency(self._t, self._q, self._w) or \
                not check_pos_def(self._q) or not check_pos_def(self._t):
            return False

        return True

    def get_parameters(self):
        """Return flat array with current matrices weights"""
        return self._parameters

    def get_bounds(self):
        """Returns two arrays with min and max of each parameter for the GA"""
        return self._bounds

    def set_bounds(self, param_bound):
        upper_bounds = [param_bound] * self._size
        lower_bounds = [-param_bound] * self._size
        self._bounds = [lower_bounds, upper_bounds]
    
    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):

        mode = 0
        if self._phase == 1:
            mode = 1
        elif self._phase == 1j:
            mode = 2

        if value is self.Mode.Probability:
            self._call = lambda data: rtbm_probability(data, self._bv, self._bh, self._t, self._w, self._q, mode)
        elif value is self.Mode.LogProbability:
            self._call = lambda data: rtbm_log_probability(data, self._bv, self._bh, self._t, self._w, self._q, mode)
        elif value is self.Mode.Expectation:
            self._call = lambda data: self._phase*hidden_expectations(data, self._bh, self._w, self._q)
        else:
            raise AssertionError('Mode %s not implemented.' % value)

    @property
    def bv(self):
        return self._bv

    @bv.setter
    def bv(self, value):
        if value.shape != self._bv.shape:
            raise AssertionError('Setting bv with wrong shape.')
        self._bv = value

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        if value.shape != self._t.shape:
            raise AssertionError('Setting t with wrong shape.')
        self._t = value

    @property
    def bh(self):
        return self._bh

    @bh.setter
    def bh(self, value):
        if value.shape != self._bh.shape:
            raise AssertionError('Setting bh with wrong shape.')
        self._bh = value

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        if value.shape != self._w.shape:
            raise AssertionError('Setting w with wrong shape.')
        self._w = value

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        if value.shape != self._q.shape:
            raise AssertionError('Setting q with wrong shape.')
        self._q = value
