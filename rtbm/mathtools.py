# -*- coding: utf-8 -*-

import numpy as np
from riemann_theta.riemann_theta import RiemannTheta
RTBM_precision= 1e-8

from numba import jit

from sympy import *
from sympy.abc import x,y

import mpmath
from numpy import frompyfunc


# TESTS different 1D thetas
test  = frompyfunc(lambda z, q: mpmath.jtheta(3, z, q), 2, 1)
testD = frompyfunc(lambda z, q: mpmath.jtheta(3, z, q, derivative=1), 2, 1)
vfunc = np.vectorize(mpmath.jtheta)


def check_normalization_consistency(t, q, w):
    c = q - np.transpose(w).dot(np.linalg.inv(t).dot(w))
    return np.all(np.linalg.eigvals(c) > 0)


def rtbm_probability(v, bv, bh, t, w, q):
    """Implements the RTBM probability"""
    return np.exp(rtbm_log_probability(v, bv, bh, t, w, q))


def rtbm_log_probability(v, bv, bh, t, w, q):
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

    ExpF = -0.5 * vTv.diagonal() - Bvv - BiTB * np.ones(v.shape[1])
  
    lnR1 = RiemannTheta.log_eval((vT.dot(w) + BhT) / (2.0j * np.pi), -q / (2.0j * np.pi), epsilon=RTBM_precision)
    lnR2 = RiemannTheta.log_eval((BhT - BtiTW) / (2.0j * np.pi), (-q + WtiTW) / (2.0j * np.pi), epsilon=RTBM_precision)

    return np.log(np.sqrt(detT / (2.0 * np.pi) ** (v.shape[0]))) + ExpF + lnR1 - lnR2



def gradient_log_theta(v, q, d):
    """ Implements the directional log gradient

        d : int for direction of gradient
    """
    Nh = q.shape[0]
    D = np.zeros(Nh)
    D[d] = 1

    R = RiemannTheta.log_eval(v / (2.0j * np.pi), -q / (2.0j * np.pi), epsilon=RTBM_precision)
    L = RiemannTheta.log_eval(v / (2.0j * np.pi), -q / (2.0j * np.pi), derivs=[D], epsilon=RTBM_precision)

    return  (- np.exp(L-R) / (2.0j * np.pi))

def gradient_log_1d_theta_phaseI(v, q, d):
    """ Implements the directional log gradient

        d : int for direction of gradient
    """
    Nh = q.shape[0]
    D = np.zeros(Nh)
    D[d] = 1

    """ Restrict to unit lattice box """

    re = np.divmod(v, q)

    R = RiemannTheta(re[1] / (2.0j * np.pi), -q / (2.0j * np.pi), epsilon=RTBM_precision)
    L = RiemannTheta(re[1] / (2.0j * np.pi), -q / (2.0j * np.pi), epsilon=RTBM_precision, derivs=[D])
    #R = test(np.pi*v/(2j*np.pi),np.exp(-1j*np.pi*q[0,0]/(2j*np.pi)))
    #L = np.pi*testD(np.pi*v/(2j*np.pi),np.exp(-1j*np.pi*q[0,0]/(2j*np.pi)))
    #R = vfunc(3,np.pi*v/(2j*np.pi),np.exp(-1j*np.pi*q[0,0]/(2j*np.pi)))
    #L = np.pi*vfunc(3,np.pi*v/(2j*np.pi),np.exp(-1j*np.pi*q[0,0]/(2j*np.pi)), derivative=1)
    
    return (-(L/R) / (2.0j * np.pi)).flatten() - re[0].flatten()



def theta_1d(v, q, d):
    """ Wraps the RT for 1d subcase with dth order directional gradient

        d : # of derivatives to take
    """
    # Cutoff if q is not positive definite
    if(np.real(q[0,0])<=0):
        q[0,0] = 1e-5
            
    if(d > 0):
        D = np.ones((1,d))
        
        R = 1.0/((2j*np.pi)**d) * RiemannTheta(v / (2.0j * np.pi), -q / (2.0j * np.pi), epsilon=RTBM_precision, derivs=D)
    else:
        
        R = RiemannTheta(v / (2.0j * np.pi), -q / (2.0j * np.pi), epsilon=RTBM_precision)
  
    return R
    
    
def logtheta_1d_phaseI(v, q, d):
    """ Wraps the RT for 1d subcase with dth order directional gradient

        d : # of derivatives to take
    """
    # Cutoff if q is not positive definite
    if(np.real(q[0,0])<=0 or np.isnan(q).any()):
        print("NaN detected or negative value in phase I: ",q)   
        q[0,0] = np.abs(q[0,0])
            
    if(d > 0):
        D = np.ones((1,d))
        
        R = -d*np.log(((2j*np.pi))) + RiemannTheta.log_eval(v / (2.0j * np.pi), -q / (2.0j * np.pi), epsilon=RTBM_precision, derivs=D)
        
    else:
        # Make NaN safe via moving to fundamental box
        re = np.divmod(v, q)
      
        e = (np.asarray(re[0])*v -0.5*q[0,0]*np.asarray(re[0])**2)[:,0]
        R = e + RiemannTheta.log_eval(re[1] / (2.0j * np.pi), -q / (2.0j * np.pi), epsilon=RTBM_precision)
       
    return R


def logtheta_1d(v, q, d):
    """ Wraps the RT for 1d subcase with dth order directional gradient

        d : # of derivatives to take
    """
    # Cutoff if q is not positive definite
    if(np.real(q[0,0])<=0 or np.isnan(q).any()):
        print("NaN detected or negative value in phase I: ",q)   
        q[0,0] = np.abs(q[0,0])
            
    if(d > 0):
        D = np.ones((1,d))
        
        R = -d*np.log(((2j*np.pi))) + RiemannTheta.log_eval(v / (2.0j * np.pi), -q / (2.0j * np.pi), epsilon=RTBM_precision, derivs=D)
        
    else:
        R = RiemannTheta.log_eval(v / (2.0j * np.pi), -q / (2.0j * np.pi), epsilon=RTBM_precision)
       
    return R

def hidden_expectations(v, bh, w, q):
    """ Implements E(h|v) for non-diagonal q

        Returns [ E(h_1|v), E(h_2|v), ... ] in vectorized form (each E is an array for the vs)
    """
    Nh = q.shape[0]

    vW = np.transpose(v).dot(w)

    E = np.zeros((Nh,v.shape[1]), dtype=complex)
    vWbhT = vW + bh.T
    
    for i in range(0, Nh):
        E[i] = gradient_log_theta(vWbhT, q, i)

    return E

def factorized_hidden_expectations(v, bh, w, q, phaseI=False):
    """ Implements E(h|v) in factorized form for q diagonal
        Note: Does not check if q is actual diagonal (for performance)

        Returns [ E(h_1|v), E(h_2|v), ... ] in vectorized form (each E is an array for the vs)
    """
    Nh = q.shape[0]

    vW = np.transpose(v).dot(w)

    E = np.zeros((Nh,v.shape[1]), dtype=complex)

    for i in range(Nh):
        O = np.matrix([[q[i, i]]], dtype=complex)
        
        # Cutoff to keep positive definite
        if(np.real(O[0,0])<=0 or np.isnan(O).any()):
            print("NaN detected or negative value: ",O)   
            O[0,0] = np.abs(O[0,0])
           
            
        if(phaseI==True):
            E[i] = gradient_log_1d_theta_phaseI(np.real((vW[:, [i]] + bh[i])), np.real(O), 0)
        else:
            E[i] = gradient_log_1d_theta_phaseII((vW[:, [i]] + bh[i]), O, 0)

    return E
