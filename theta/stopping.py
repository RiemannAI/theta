# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod


class stopping(object):
    """ Abstract class for stopping condition """
    __metaclass__ = ABCMeta

    @abstractmethod
    def do_stop(self, v):
        pass


class earlystop(stopping):
    """A bare bone early stopping class"""
    def __init__(self, delta=10):
        self.delta = delta

    def do_stop(self, v):
        """implemet the condition"""
        if len(v) < self.delta:
            return False

        # stops if validation in the delta window grows
        if v[-1]-v[-self.delta] > 0:
            return True

        return False