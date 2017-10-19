#!/usr/bin/env sage
# -*- coding: utf-8 -*-

import numpy as np


def logarithmic(x, *y):
    """ Logarithmic total cost """
    return -np.sum(np.log(x))


def mse(x, y):
    """ Mean squared error """
    return np.mean((x-y)**2)
