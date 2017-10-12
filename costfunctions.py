#!/usr/bin/env sage
# -*- coding: utf-8 -*-

import numpy as np


def logarithmic(x, *y):
    """ Logarithmic total cost """
    try:
        res = -np.sum(np.log(x))
        if np.isnan(res):
            res = np.inf
    except:
        res = np.inf
    return res


def mse(x, y):
    """ Mean squared error """
    return np.mean((x-y)**2)
