# -*- coding: utf-8 -*-
import numpy as np


class Model(object):
    """The model class which holds layer for building mixture models and theta neural networks.

    Example:
        Here a short example showing how to allocate the Model class::

            from theta.theta import Model
            from theta.layers import ThetaUnitLayer, AddNormLayer
            m = Model()  # allocates a RTBM with Nv=1 and Nh=2
            m.add(ThetaUnitLayer(1,2))
            m.add(NormAddLayer(2,1))
    """

    def __init__(self):
        self._layers = []
        self._Np = 0
        self._lower_bounds = None
        self._upper_bounds = None

    def __call__(self, data):
        """Evaluates the model for a given data array"""
        return self.feed_through(data)

    def feed_through(self, X, grad_calc=False):
        """Feeds the input X through all layers.
        This method is equivalent in calling ``()`` operator.

        Args:
            X (numpy.array): the feedin data, shape (Nv, Ndata).
            grad_calc (bool): if True stores useful data for backpropagation.

        Returns:
            numpy.array: predictions for the Model.
        """
        x = X
        for L in self._layers:
            x = L.feedin(x, grad_calc)
        return x

    def add(self, layer):
        """ Add layer to the model instance.

        Args:
             layer (theta.layers): any layer implemented in theta.layers (Layers_).

        Warning:
            Input of layer does not match output of previous layer!
        """
        if len(self._layers) == 0 or self._layers[-1].get_Nout() == layer.get_Nin():
            # Add the layer
            self._layers.append(layer)

            # Increase the parameter counter
            self._Np = self._Np + layer.size()

            # Re-compute bound vectors
            self.build_bounds()

        else:
            print("Input of layer does not match output of previous layer! => Add ignored")

    def backprop(self, E):
        """Evaluates and stores the gradients for backpropagation.

        Args:
            E (numpy.array): the error for backpropagation.

        Returns:
            numpy.array: the updated error function.
        """
        e = E

        for i in reversed(range(len(self._layers))):
            e = self._layers[i].backprop(e)
        return e

    def size(self):
        """
        Returns:
            int: the size of the RTBM.
        """
        return self._Np

    def get_parameters(self):
        """Collects all parameters and returns a flat array.

        Returns:
            numpy.array: flat array with current matrices weights.
        """
        return np.concatenate([L.get_parameters() for L in self._layers]).ravel()

    def get_gradients(self):
        """Collects all gradients and returns a flat array.
        Returns:
            numpy array: flat array with calculated gradients.
        """
        return np.concatenate([L.get_gradients() for L in self._layers]).ravel()

    def get_layer(self, N):
        """
        Args:
            N (int): the layer number.
        Returns:
            theta.layers: returns the N-th layer stored in the model
        """
        if N > len(self._layers):
            print("Layer does not exist")
            return None
        else:
            return self._layers[N-1]

    def build_bounds(self):
        """
        Returns:
            list of numpy.array: collects the bounds of the individual layers
        """
        A_L = []
        A_U = []

        for L in self._layers:
            bl, bu = L.get_bounds()
            A_L.append(bl)
            A_U.append(bu)

        lower_bounds = np.concatenate(A_L).tolist()
        upper_bounds = np.concatenate(A_U).tolist()
        self._bounds = [lower_bounds,upper_bounds]

    def set_parameters(self, params):
        """Assigns a flat array of parameters to the Model.

        Args:
            params (numpy.array): list of parameters assigned to the Model.

        Returns:
            bool: True if assignment is successful, False otherwise.
        """
        Nt = 0

        for L in self._layers:

            if not L.set_parameters(params[Nt:Nt+L.size()]):
                return False
            Nt += L.size()

        return True

    def get_bounds(self):
        """
        Returns:
            list of numpy.array: two arrays with min and max of each parameter of all layers.
        """
        return self._bounds

    def set_bound(self, bound):
        """Sets the parameter bound for each parameter.

        Args:
            bound (float): the maximum absolute value for parameter variation.
        """
        for L in self._layers:
            L.set_bounds(bound)

    def gradient_check(self, g, x, epsilon):
        """Performs numerical check of gth gradient.

        Args:
            g (int): id of gradient to check
            x (numpy.array): input data shape (Ninput, Ndata).
            epsilon (float): infinitesimal variation of parameter

        Returns:
            floats: the numerical and analytical gradients.
        """
        print("I: ",x)

        # Prepare parameters
        W = self.get_parameters()
        print("P: ", W)
        O = self.feed_through(x, True)
        print("O: ",O)
        print("=======")

        # Calc backprop derivative
        self.backprop(np.ones(O.shape))
        G = self.get_gradients()

        Wp = W.copy()
        Wm = W.copy()

        Wp[g] = Wp[g] + epsilon
        Wm[g] = Wm[g] - epsilon

        # Calc numerical derivative
        self.set_parameters(Wp)
        P = self.feed_through(x)
        self.set_parameters(Wm)
        M = self.feed_through(x)

        D = (P-M)/(2*epsilon)

        res_num = np.mean(D, axis=1)
        print(g,"th (mean) numerical gradient: ")
        print(res_num[0])

        # Calc backprop derivative
        self.set_parameters(W)
        self.backprop(np.ones(O.shape))
        G = self.get_gradients()

        res_back = G[g]
        print(g,"th (mean) backprop gradient: ")
        print(res_back)

        return res_num, res_back
