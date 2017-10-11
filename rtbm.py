#!/usr/bin/env sage
# -*- coding: utf-8 -*-

import numpy as np
import copy
from abelfunctions import RiemannTheta

RTBM_precision = 1e-16

class RTBM(object):
    """This class implements the Riemann Theta Boltzmann Machine"""

    def __init__(self, visible_units, hidden_units):
        """Setup operators for BM based on the number of visible and hidden units"""
        self._bv = np.zeros([visible_units, 1], dtype=complex)
        self._t = np.ones([visible_units, visible_units], dtype=complex)
        self._bh = np.zeros([hidden_units, 1], dtype=complex)
        self._w = np.zeros([visible_units, hidden_units], dtype=complex)
        self._q = np.zeros([hidden_units, hidden_units], dtype=complex)

        self._Nv = visible_units
        self._Nh = hidden_units
        
        # Set default parameter bound value
        self.param_bound = 10

        # Populate with non-zero parameters
        self.assign_params(np.array([1E-5]*self.size()))
        
    def __call__(self, data):
        """Evaluates the RTBM instance for a given data array"""
        return probability(data, self._bv, self._bh, self._t, self._w, self._q)

    def copy(self):
        return copy.deepcopy(self)

    def size(self):
        """Get size of RTBM"""
        return 2*self._Nv + 2*self._Nh + self._Nv*self._Nh

    def get_bounds(self):
        """Returns two arrays with min and max of each parameter for the GA"""
        lower_bounds = [-self._param_bound for i in range(self.size())]
        upper_bounds = [ self._param_bound for i in range(self.size())]

        # set T positive
        if self._bv.shape[0] == 1:
            index = self._bv.shape[0]
            lower_bounds[index:index+self._t.shape[0]] = [1E-5]*self._t.shape[0]

        # set Q positive
        index = self.size()-self._q.shape[0]
        lower_bounds[index:] = [1E-5]*self._q.shape[0]

        return lower_bounds, upper_bounds
    
    def assign_params(self, params):
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

    def get_params(self):
        """Return flat array with current matrices weights"""
        params = np.concatenate([self._bv.flatten(), self._t.diagonal(), self._bh.flatten(),
                   self._w.flatten(), self._q.diagonal()])
        return params

    def random_init(self, Tmax=2, Qmax=5, Wmax=2):
        """ Initalizes the RTBM parameters uniform random
        (the Bs are kept @ 0)
        """
        # Init random diagonal pos. def. 
        self._t = np.diag(np.random.uniform(0.01,Tmax,self._Nv)).astype(complex)
        self._q = np.diag(np.random.uniform(0.01,Qmax,self._Nh)).astype(complex)
        
        # Init random
        self._w = np.random.uniform(-Wmax,Wmax,(self._Nv,self._Nh))
        
        while(checkNormalizationConsistency(self._t,self._q,self._w) == False):
            self._w = np.random.uniform(-Wmax,Wmax,(self._Nv,self._Nh))
            
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

    R1 = RiemannTheta((vT.dot(w) + BhT) / (2.0j * np.pi), -q / (2.0j * np.pi), prec=RTBM_precision)
    R2 = RiemannTheta((BhT - BtiTW) / (2.0j * np.pi), (-q + WtiTW) / (2.0j * np.pi), prec=RTBM_precision)

    return np.sqrt(detT / (2.0 * np.pi) ** (v.shape[0])) * ExpF * R1/R2


def gradientLogTheta(v,q,d):
    """ Implements the directional log gradient 
        
        d : int for direction of gradient
    """
    Nh = q.shape[0]
    D = np.zeros(Nh)
    D[d] = 1
    
    R = RiemannTheta(v, q, prec=RTBM_precision)
    L = RiemannTheta(v, q, prec=RTBM_precision, derivs=[D])
    
    return L/R


def factorizedHiddenExpectation(v,bh,w,q):
    """ Implements E(h|v) in factorized form for q diagonal 
        Note: Does not check if q is actual diagonal (for performance)
        
        Returns [ E(h_1|v), E(h_2|v), ... ] in vectorized form (each E is an array for the vs)
    """ 
    Nh = q.shape[0]
    
    vW = np.transpose(v).dot(w)
    
    E = []
    
    for i in range(0,Nh):
        O = np.matrix([[q[i,i]]], dtype=np.complex)
        E.append( -gradientLogTheta(vW[:,[i]]+bh[i],O,0) )
    
    return E
    
    
def checkNormalizationConsistency(T,Q,W):
    C = Q - np.transpose(W).dot(np.linalg.inv(T).dot(W))
    return np.all(np.linalg.eigvals(C) > 0)