from layers import Layer 

import numpy as np

class Model(object):
    
    def __init__(self):
        """ Init
        """
        self._layers = [] 
        
    def add(self, L):
        
        if len(self._layers) == 0 or self._layers[-1].get_Nout()==L.get_Nin():
            self._layers.append(L)
        else: 
            print("Input of layer does not match output of previous layer! => Add ignored")

    def feedthrough(self, X):
        """ Feeds the input X through all layers 
            Vectorized
        """ 

        x = X

        for L in self._layers:
            
            x = L.feedin(x)

        return x

    def get_parameters(self):
        """ Collects all parameters and returns a flat array """
        R = []
        for L in self._layers:
            R.append(L.get_parameters())
            
        return np.concatenate(R)

    def set_parameters(self,P):
        """ Set to the given parameters """
        Nt = 0
        
        for L in self._layers:
            Np = L.get_Nparameters()
            
            L.set_parameters(P[Nt:Nt+Np])
        
            Nt = Nt + Np    

    def train(self,X,Y):
        """ Trains the model with the set cost function """
        #...

    def predict(self,X):
        """ Performs prediction with the trained model """

        return feedthrough(X)
