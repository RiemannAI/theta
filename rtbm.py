#!/usr/bin/env sage
# -*- coding: utf-8 -*-

import numpy as np

from mathtools import rtbm_probability, check_normalization_consistency, \
    factorized_hidden_expectation, rtbm_log_probability


class AssignError(Exception):
    pass


class RTBM(object):
    """This class implements the Riemann Theta Boltzmann Machine"""
    class Mode:
        Probability = 0
        LogProbability = 1
        Expectation = 2

    def __init__(self, visible_units, hidden_units, mode=Mode.Probability, param_bound=0.5):
        """Setup operators for BM based on the number of visible and hidden units"""
        self._Nv = visible_units
        self._Nh = hidden_units
        self._bv = np.zeros([visible_units, 1], dtype=complex)
        self._t = np.ones([visible_units, visible_units], dtype=complex)
        self._bh = np.zeros([hidden_units, 1], dtype=complex)
        self._w = np.zeros([visible_units, hidden_units], dtype=complex)
        self._q = np.zeros([hidden_units, hidden_units], dtype=complex)
        self._a_size = (self._Nv+self._Nh)**2
        self._a_shape = ((self._Nv+self._Nh), (self._Nv+self._Nh))
        self._size = self._Nv + self._Nh + self._a_size
        self._mode = None
        self._call = None

        # Populate with random parameters
        self._parameters = np.random.uniform(-param_bound, param_bound, self._size)
        self.set_parameters(self._parameters)

        # set operation mode
        self.mode = mode

    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):
        if value is self.Mode.Probability:
            self._call = lambda data: rtbm_probability(data, self._bv, self._bh, self._t, self._w, self._q)
        elif value is self.Mode.LogProbability:
            self._call = lambda data: rtbm_log_probability(data, self._bv, self._bh, self._t, self._w, self._q)
        elif value is self.Mode.Expectation:
            self._call = lambda data: factorized_hidden_expectation(data, self._bh, self._w, self._q)
        else:
            raise AssertionError('Mode %s not implemented.' % value)

    def __call__(self, data):
        """Evaluates the RTBM instance for a given data array"""
        return self._call(data)

    def size(self):
        """Get size of RTBM"""
        return self._size

    def set_parameters(self, params):
        """Assigns a flat array of parameters to the RTBM matrices"""
        if len(params) != self._size:
            raise Exception('Size does no match.')

        self._parameters = params
        x = params[0:self._a_size].reshape(self._a_shape)
        a = np.transpose(x).dot(x)
        self._q = a[:self._Nh,:self._Nh]
        self._t = a[self._Nh:self._Nh+self._Nv,self._Nh:]
        self._w = a[self._Nh:,:self._Nh]
        self._bv = params[self._a_size:self._a_size+self._Nv].reshape(self._bv.shape)
        self._bh = params[-self._Nh:].reshape(self._bh.shape)

        if not check_normalization_consistency(self._t, self._q, self._w):
            raise AssignError('not positive random initialization')

    def get_parameters(self):
        """Return flat array with current matrices weights"""
        return self._parameters

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
