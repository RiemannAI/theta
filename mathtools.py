#!/usr/bin/env sage
# -*- coding: utf-8 -*-

import numpy as np
from abelfunctions import RiemannTheta
RTBM_precision= 1e-16


def check_normalization_consistency(t, q, w):
    c = q - np.transpose(w).dot(np.linalg.inv(t).dot(w))
    return np.all(np.linalg.eigvals(c) > 0)


def rtbm_probability(v, bv, bh, t, w, q):
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

    return np.sqrt(detT / (2.0 * np.pi) ** (v.shape[0])) * ExpF * R1 / R2


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

    R1 = RiemannTheta((vT.dot(w) + BhT) / (2.0j * np.pi), -q / (2.0j * np.pi), prec=RTBM_precision)
    R2 = RiemannTheta((BhT - BtiTW) / (2.0j * np.pi), (-q + WtiTW) / (2.0j * np.pi), prec=RTBM_precision)

    return np.log(np.sqrt(detT / (2.0 * np.pi) ** (v.shape[0]))) + ExpF + np.log(R1) - np.log(R2)


def gradient_log_theta(v, q, d):
    """ Implements the directional log gradient

        d : int for direction of gradient
    """
    Nh = q.shape[0]
    D = np.zeros(Nh)
    D[d] = 1

    R = RiemannTheta(v / (2.0j * np.pi), -q / (2.0j * np.pi), prec=RTBM_precision)
    L = RiemannTheta(v / (2.0j * np.pi), -q / (2.0j * np.pi), prec=RTBM_precision, derivs=[D])

    """ ToDo: Check if not some factor is missing ... """

    return -L / R / (2.0j * np.pi)


def factorized_hidden_expectation(v, bh, w, q):
    """ Implements E(h|v) in factorized form for q diagonal
        Note: Does not check if q is actual diagonal (for performance)

        Returns [ E(h_1|v), E(h_2|v), ... ] in vectorized form (each E is an array for the vs)
    """
    Nh = q.shape[0]

    vW = np.transpose(v).dot(w)

    E = np.zeros(Nh)

    for i in range(Nh):
        O = np.matrix([[q[i, i]]], dtype=np.complex)

        E[i] = gradient_log_theta((vW[:, [i]] + bh[i]), O, 0)

    return E