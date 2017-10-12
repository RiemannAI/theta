
import numpy as np

""" Logarithmic total cost """
def Logarithmic(X,Y):
    try:
        res = -np.sum(np.log(X))
        if np.isnan(res): res = np.inf
    except:
        res = np.inf
        
    return res
    
""" Mean squared error """
def MSE(X,Y):
    return np.mean((X-Y)**2)
