# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class stopping(object):
    """ Abstract class for stopping condition """
    __metaclass__ = ABCMeta

    @abstractmethod
    def do_stop(self, v):
        pass


class earlystop(stopping):
    """A simple implementation of early stopping.
    The validation loss function is monitored if it increases
    after `delta` iterations the stop signal is send to the minimizer.

    Args:
        delta (int): the number of iterations where the validation is monitored.
    """
    def __init__(self, delta=10):
        self.delta = delta

    def do_stop(self, v):
        """Function which tests if the stop condition is reached.

        Args:
            v (numpy.array): history of the validation loss function.

        Returns:
            bool: True if the validation loss is growing in the delta window, False elsewhere.
        """
        if len(v) < self.delta:
            return False

        # stops if validation in the delta window grows
        if v[-1]-v[-self.delta] > 0:
            return True

        return False