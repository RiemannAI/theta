#!/usr/bin/env sage
# -*- coding: utf-8 -*-

import numpy as np
from mathtools import rtbm_probability, check_normalization_consistency


class AssignError(Exception):
    pass


class RTBM(object):
    """This class implements the Riemann Theta Boltzmann Machine"""

    def __init__(self, visible_units, hidden_units):
        """Setup operators for BM based on the number of visible and hidden units"""
        self._Nv = visible_units
        self._Nh = hidden_units
        self._bv = np.zeros([visible_units, 1], dtype=complex)
        self._t = np.ones([visible_units, visible_units], dtype=complex)
        self._bh = np.zeros([hidden_units, 1], dtype=complex)
        self._w = np.zeros([visible_units, hidden_units], dtype=complex)
        self._q = np.zeros([hidden_units, hidden_units], dtype=complex)
        
        # Set default parameter bound value
        self._param_bound = 10

        # Populate with random parameters
        self.random_init()
        
    def __call__(self, data):
        """Evaluates the RTBM instance for a given data array"""
        return rtbm_probability(data, self._bv, self._bh, self._t, self._w, self._q)

    def size(self):
        """Get size of RTBM"""
        return 2*self._Nv + 2*self._Nh + self._Nv*self._Nh

    def get_bounds(self):
        """Returns two arrays with min and max of each parameter for the GA"""
        lower_bounds = [-self._param_bound for _ in range(self.size())]
        upper_bounds = [ self._param_bound for _ in range(self.size())]

        # set T positive
        if self._bv.shape[0] == 1:
            index = self._bv.shape[0]
            lower_bounds[index:index+self._t.shape[0]] = [1E-5]*self._t.shape[0]

        # set Q positive
        index = self.size()-self._q.shape[0]
        lower_bounds[index:] = [1E-5]*self._q.shape[0]

        return lower_bounds, upper_bounds

    def set_parameters(self, params):
        """Assigns a flat array of parameters to the RTBM matrices"""
        if len(params) != self.size():
            raise Exception('Size does no match.')

        index = self._bv.shape[0]
        bv = params[0:index].reshape(self._bv.shape)

        t = np.diag(params[index:index+self._t.shape[0]])
        index += self._t.shape[0]

        bh = params[index:index+self._bh.shape[0]].reshape(self._bh.shape)
        index += self._bh.shape[0]

        w = params[index:index+self._w.size].reshape(self._w.shape)
        index += self._w.size

        q = np.diag(params[index:index+self._q.shape[0]])
        
        # Only keep if consistent solution
        # Temporary work-around
        if check_normalization_consistency(t, q, w):
            self._w = w
            self._q = q
            self._t = t
            self._bv = bv
            self._bh = bh
        else:
            raise AssignError('RTBM assign_params: check normalization consistency failed')

    def get_parameters(self):
        """Return flat array with current matrices weights"""
        params = np.concatenate([self._bv.flatten(), self._t.diagonal(), self._bh.flatten(),
                                 self._w.flatten(), self._q.diagonal()])
        return params
        
    def random_init(self, t_max=2, q_max=5, w_max=2):
        """ Initalizes the RTBM parameters uniform random
        (the Bs are kept @ 0)"""

        # Init random diagonal pos. def. 
        self._t = np.diag(np.random.uniform(0.01, t_max, self._Nv)).astype(complex)
        self._q = np.diag(np.random.uniform(0.01, q_max, self._Nh)).astype(complex)
        
        # Init random
        self._w = np.random.uniform(-w_max, w_max, (self._Nv, self._Nh)).astype(complex)
        
        while not check_normalization_consistency(self._t, self._q, self._w):
            self._w = np.random.uniform(-w_max, w_max, (self._Nv, self._Nh))

    @property
    def param_bound(self):
        return self._param_bound

    @param_bound.setter
    def param_bound(self, bound):
        """ Set the global bound for the parameters """
        if np.isnan(bound) or np.isinf(bound):
            raise AssertionError('Bound is nan or inf')
        self._param_bound = bound

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
