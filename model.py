from layers import Layer 


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

    def get_parameters():
        """ Collects all parameters and returns a flat array """
        #...

    def set_parameters(P):
        """ Sets the new parameters """
        #...


    def set_costfunction(C):
        """ Sets the cost function to be used """
        #...
    

    def train(X,Y):
       """ Trains the model with the set costfunction"""
       #...

    def predict(X):
        """ Performs prediction with the trained model """

        return feedthrough(X)
