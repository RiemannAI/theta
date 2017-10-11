
import numpy as np

def logarithmic(X,Y):
    try:
        res = -np.sum(np.log(X))
        if np.isnan(res): res = np.inf
    except:
        res = np.inf
    
    