#!/usr/bin/env sage
# -*- coding: utf-8 -*-

import numpy as np
from abelfunctions import RiemannTheta


class RTBM(object):
    """This class implements the Riemann Theta Boltzmann Machine"""

    def __init__(self, visible_units, hidden_units):
        """Setup operators for BM based on the number of visible and hidden units"""
        self._bv = np.zeros([visible_units, 1], dtype=complex)
        self._t = np.ones([visible_units, visible_units], dtype=complex)
        self._bh = np.zeros([hidden_units, 1], dtype=complex)
        self._w = np.zeros([visible_units, hidden_units], dtype=complex)
        self._q = np.zeros([hidden_units, hidden_units], dtype=complex)

    def __call__(self, data):
        """Evaluates the RTBM instance for a given data array"""
        return probability(data, self._bv, self._bh, self._t, self._w, self._q)

    def size(self):
        """Get size of RTBM"""
        return self._bv.shape[0] + self._t.shape[0] + self._bh.shape[0] + self._w.size + self._q.shape[0]

    def get_bounds(self, absmax=10):
        """Returns two arrays with min and max of each parameter for the GA"""
        lower_bounds = [-absmax for i in range(self.size())]
        upper_bounds = [ absmax for i in range(self.size())]

        # set T positive
        if self._bv.shape[0] == 1:
            index = self._bv.shape[0]
            lower_bounds[index:index+self._t.shape[0]] = [1E-5]*self._t.shape[0]

        # set Q positive
        index = self.size()-self._q.shape[0]
        lower_bounds[index:] = [1E-5]*self._q.shape[0]

        return lower_bounds, upper_bounds

    def assign(self, params):
        """Assigns a flat array of parameters to the RTBM matrices"""
        if len(params) != self.size():
            raise Exception('Size does no match.')

        index = self._bv.shape[0]
        self._bv = params[0:index].reshape(self._bv.shape)

        np.fill_diagonal(self._t, params[index:index+self._t.shape[0]])
        index += self._t.shape[0]

        self._bh = params[index:index+self._bh.shape[0]].reshape(self._bh.shape)
        index += self._bh.shape[0]

        self._w = params[index:index+self._w.size].reshape(self._w.shape)
        index += self._w.size

        np.fill_diagonal(self._q, params[index:index+self._q.shape[0]])

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


def probability(v, bv, bh, t, w, q):
    """Implements the RTBM probability"""
    detT = np.linalg.det(t)
    invT = np.linalg.inv(t)
    vT = v.T
    vTv = np.dot(np.dot(vT, t), v)
    BvT = bv.T
    BhT = bh.T
    Bvv = np.dot(BvT, v)
    BiTB = np.dot(np.dot(BvT, invT), bv)
    BtiTW = np.dot(np.dot(BvT, invT), w)
    WtiTW = np.dot(np.dot(w.T, invT), w)

    ExpF = np.exp(-0.5 * vTv.diagonal() - Bvv - BiTB * np.ones(v.shape[1]))

    R1 = RiemannTheta((vT.dot(w) + BhT) / (2.0j * np.pi), -q / (2.0j * np.pi), prec=1e-16)
    R2 = RiemannTheta((BhT - BtiTW) / (2.0j * np.pi), (-q + WtiTW) / (2.0j * np.pi), prec=1e-16)

    return np.sqrt(detT / (2.0 * np.pi) ** (v.shape[0])) * ExpF * R1/R2