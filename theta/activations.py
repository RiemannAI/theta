# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import numpy as np


class actfunc(object):
    """ Abstract class for cost functions """
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def activation(self, X):
        pass
    
    @abstractmethod
    def gradient(self, X):
        pass


class linear(actfunc):
    """A linear pass through."""
    
    @staticmethod
    def activation(x):
        """Evaluates the activation function.

        Args:
            x (numpy.array): the input data.

        Returns:
            numpy.array: the activation function evaluation.
        """
        return x

    @staticmethod
    def gradient(x):
        """Evaluates the gradient of the activation function.

        Args:
            x (numpy.array): the input data.

        Returns:
            numpy.array: the gradient of the activation function.
        """
        return np.ones(x.shape)


class sigmoid(actfunc):
    """The sigmoid activation."""

    @staticmethod
    def activation(x):
        """Evaluates the activation function.

        Args:
            x (numpy.array): the input data.

        Returns:
            numpy.array: the activation function evaluation.
        """
        return 1.0/(1+np.exp(-x))
    
    @staticmethod
    def gradient(x):
        """Evaluates the gradient of the activation function.

        Args:
            x (numpy.array): the input data.

        Returns:
            numpy.array: the gradient of the activation function.
        """
        e = np.exp(x)
        return e/((1+e)**2)


class tanh(actfunc):
    """ The tanh activation."""

    @staticmethod
    def activation(x):
        """Evaluates the activation function.

        Args:
            x (numpy.array): the input data.

        Returns:
            numpy.array: the activation function evaluation.
        """
        return np.tanh(x)

    @staticmethod
    def gradient(x):
        """Evaluates the gradient of the activation function.

        Args:
            x (numpy.array): the input data.

        Returns:
            numpy.array: the gradient of the activation function.
        """
        return 1.0/(np.cosh(x)**2)