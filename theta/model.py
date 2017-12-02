# -*- coding: utf-8 -*-
import numpy as np


class Model(object):

    def __init__(self):
        """ Initialize attributes """
        self._layers = []
        self._Np = 0
        self._lower_bounds = None
        self._upper_bounds = None

    def add(self, layer):
        """ Add layer to model """
        if len(self._layers) == 0 or self._layers[-1].get_Nout() == layer.get_Nin():
            # Add the layer
            self._layers.append(layer)

            # Increase the parameter counter
            self._Np = self._Np + layer.size()

            # Re-compute bound vectors
            self.build_bounds()

        else:
            print("Input of layer does not match output of previous layer! => Add ignored")

    def feed_through(self, X, grad_calc=False):
        """ Feeds the input X through all layers Vectorized """
        x = X
        for L in self._layers:
            x = L.feedin(x, grad_calc)
        return x

    def backprop(self, E):
        """ Backpropagates the error E through the network """
        e = E

        for i in reversed(range(len(self._layers))):
            e = self._layers[i].backprop(e)
        return e


    def __call__(self, data):
        """ Evaluates the model for a given data array """
        return self.feed_through(data)

    def size(self):
        return self._Np

    def get_parameters(self):
        """ Collects all parameters and returns a flat array """
        return np.concatenate([L.get_parameters() for L in self._layers]).ravel()

    def get_gradients(self):
        """ Collects all gradients and returns a flat array """
        return np.concatenate([L.get_gradients() for L in self._layers]).ravel()

    def get_layer(self, N):
        if(N > len(self._layers)):
            print("Layer does not exist")
        else:
            return self._layers[N-1]

    def build_bounds(self):
        """ Collects the bounds of the individual layers """
        A_L = []
        A_U = []

        for L in self._layers:
            bl, bu = L.get_bounds()
            A_L.append(bl)
            A_U.append(bu)

        lower_bounds = np.concatenate(A_L).tolist()
        upper_bounds = np.concatenate(A_U).tolist()
        self._bounds = [lower_bounds,upper_bounds]

    def get_bounds(self):
        """ Returns the combined bounds of all layers """

        return self._bounds

    def set_parameters(self, params):
        """ Set to the given parameters """
        Nt = 0

        for L in self._layers:

            if not L.set_parameters(params[Nt:Nt+L.size()]):
                return False
            Nt += L.size()

        return True

    def predict(self, x):
        """ Performs prediction with the trained model """
        return self.feed_through(x)


    def set_bound(self, bound):
        for L in self._layers:
            L.set_bounds(bound)


    def gradient_check(self, g, x, epsilon):
        """ Performs numerical check of gth gradient
            g      : id of gradient to check
            x      : input data
            epsilon: infinitesimal variation of parameter
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
        print g,"th (mean) numerical gradient: "
        print(res_num[0])

        # Calc backprop derivative
        self.set_parameters(W)

        
        self.backprop(np.ones(O.shape))
        G = self.get_gradients()

        res_back = G[g]
        print g,"th (mean) backprop gradient: "
        print(res_back)

        return res_num, res_back
