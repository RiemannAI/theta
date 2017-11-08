# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np
from mathtools import factorized_hidden_expectations,theta_1d,logtheta_1d_phaseI, logtheta_1d

import time

from multiprocessing import Pool
pool = Pool()


class Layer(object):
    """ Abstract class for a layer of a deep network """
    __metaclass__ = ABCMeta

    @abstractmethod
    def feedin(self, X, *grad_calc):
        pass
    
    @abstractmethod
    def get_parameters(self):
        pass
    
    @abstractmethod
    def set_parameters(self, params):
        pass
    
    @abstractmethod
    def size(self):
        pass
    
    @abstractmethod
    def get_bounds(self):
        pass

    def get_Nin(self):
        return self._Nin

    def get_Nout(self):
        return self._Nout

    def size(self):
        """ Returns total # parameters """
        return self._Np
    
    
class NormAddLayer(Layer):
    """ Linearly combines inputs with outputs normalized by sum of weights """
    """ (no bias) """
    
    def __init__(self, Nin, Nout, param_bound=10):
        self._Nin = Nin
        self._Nout = Nout
        self._param_bound = param_bound
        
        # Parameter init
        self._w = np.random.uniform(-1, 1,(Nout,Nin)).astype(complex)
        
        self._Np = self._Nout*self._Nin
   
    def get_parameters(self):
        """ Returns the parameters as a flat array 
            [w]
        """

        return self._w.flatten()
  
    def set_parameters(self, P):
        """ Set the matrices from flat input array P 
            P = [w]
        """
        
        self._w = P.reshape(self._w.shape)
    
    def get_bounds(self):
        """Returns two arrays with min and max of each parameter for the GA"""
        self._lower_bounds = [-self._param_bound for _ in range(self._Np)]
        self._upper_bounds = [ self._param_bound for _ in range(self._Np)]

        return [self._lower_bounds, self._upper_bounds]
    
    def feedin(self, X, *grad_calc):
        """ Feeds in the data X and returns the output of the layer 
            Note: Vectorized 
        """
    
        S = np.sum(self._w,axis=1)
        O = self._w.dot(X)
       
        return np.divide(O, S[:, np.newaxis])
    
   
    
class Linear(Layer):
    """ Linear layer """
    def __init__(self, Nin, Nout, Wmax=1, Bmax=1, paramBound=10):
        self._Nin  = Nin
        self._Nout = Nout
        self._Np = Nin*Nout+Nout
         
        # Set bounds
        self._lower_bounds = [-paramBound for _ in range(self._Np)]
        self._upper_bounds = [ paramBound for _ in range(self._Np)]    
            
        # Parameter init
        self._w = np.random.uniform(-Wmax, Wmax,(Nout,Nin)).astype(float)
        self._b = np.random.uniform(-Bmax, Bmax,(Nout,1)).astype(float)
        
        
    def get_parameters(self):
        """ Returns the parameters as a flat array 
            [b,w]
        """

        return np.concatenate([self._b.flatten(),self._w.flatten()])

    def get_gradients(self):
        """ Returns gradients as a flat array 
            [b,w]
        """
        return np.concatenate([self._gradB.flatten(),self._gradW.flatten()])
    
    
    def feedin(self, X, grad_calc=False):
        """ Feeds in the data X and returns the output of the layer 
            Note: Vectorized 
        """
        if(grad_calc==True):
            self._X = X
        
        return self._w.dot(X)+self._b
    
    def backprop(self, E):
        """ Propagates the error E through the layer and stores gradient """
       
        # Mean bias gradient
        self._gradB = np.mean(E, axis=1,keepdims=True)
      
        # Mean weight gradient
        self._gradW = E.dot(self._X.T)/self._X.shape[1]
        
        # Propagate error
        return self._w.T.dot(E)
    
    
    def set_parameters(self, params):
        """ Set the matrices from flat input array P 
            P = [b,w]
        """
        index = 0
        
        self._b = params[index:index+self._b.shape[0]].reshape(self._b.shape)
        index += self._b.shape[0]

        self._w = params[index:index+self._w.size].reshape(self._w.shape)
       
    
    def get_bounds(self):
        """Returns two arrays with min and max of each parameter for the GA"""
        return [self._lower_bounds, self._upper_bounds]
    
    


class SoftMaxLayer(Layer):
    """ A layer to perform the softmax operation """
    def __init__(self, Nin):
        self._Nin  = Nin
        self._Nout = Nin
        self._Np = 0
        self._param_bound = 0
        
    def get_parameters(self):
        """ Returns the parameters as a flat array 
            []
        """

        return np.empty(0)
    
    def set_parameters(self, params):
        return
        
    
    def get_bounds(self):
        """Returns two arrays with min and max of each parameter for the GA"""
        self._lower_bounds = []
        self._upper_bounds = []

        return [self._lower_bounds, self._upper_bounds]
    
    def feedin(self, X, *grad_calc):
        """ Feeds in the data X and returns the output of the layer 
            Note: Vectorized 
        """
        
        E = np.exp(X)
        S = np.sum(E,axis=0) 
       
        return np.divide(E, S[np.newaxis,:])
    
    
class MaxPosLayer(Layer):
    def __init__(self, Nin, startPos=0):
        self._Nin = Nin
        self._Nout = 1
        self._Np = 0
        self._param_bound = 0
        self._startPos = startPos
        
    def get_parameters(self):
        """ Returns the parameters as a flat array 
            []
        """

        return np.empty(0)
    
    def set_parameters(self, params):
        return
        
    
    def get_bounds(self):
        """Returns two arrays with min and max of each parameter for the GA"""
        self._lower_bounds = []
        self._upper_bounds = []

        return [self._lower_bounds, self._upper_bounds]
    
    def feedin(self, X, *grad_calc):
        """ Feeds in the data X and returns the output of the layer 
            Note: Vectorized 
        """
        
        return np.argmax(X,axis=0)+self._startPos
    
 
        
class DiagExpectationUnitLayer(Layer):
    """ A layer of log-gradient theta units """

    def __init__(self, Nin, Nout, Wmax=1,Bmax=1,Qmax=10, phase=1, paramBound=10):
        self._Nin = Nin
        self._Nout = Nout
        self._phase = phase
        
        dtype = complex
        
        # Parameter init
        if(phase == 1):
            dtype = float
            
        self._bh = phase*np.random.uniform(-Bmax, Bmax,(Nout,1)).astype(dtype)
        self._w = phase*np.random.uniform(-Wmax, Wmax,(Nin,Nout)).astype(dtype)
        
        self._q = Qmax*np.diag(np.random.rand(Nout)).astype(complex)
     
        self._Np = 2*self._Nout+self._Nout*self._Nin
        
        # Set B bounds
        self._lower_bounds = [-paramBound for _ in range(self._Np)]
        self._upper_bounds = [ paramBound for _ in range(self._Np)]
    
        # Set W bounds
        index = self._Np-self._q.shape[0]-self._w.shape[0]
        self._lower_bounds[index:] = [-paramBound]*self._w.shape[0]
        self._upper_bounds[index:] = [paramBound]*self._w.shape[0]
        
        # set q bounds
        index = self._Np-self._q.shape[0]
        self._lower_bounds[index:] = [1E-5]*self._q.shape[0]
        self._upper_bounds[index:] = [paramBound]*self._q.shape[0]
        
    def feedin(self, X, grad_calc=False):
        """ Feeds in the data X and returns the output of the layer 
            Note: Vectorized 
        """
        
        if(grad_calc==True):
            self._X = X
        
        if(self._phase==1):
            return 1.0/self._phase*np.array(factorized_hidden_expectations(X,self._bh,self._w,self._q, True))
        else:
            return 1.0/self._phase*np.array(factorized_hidden_expectations(X,self._bh,self._w,self._q, False))

    def get_parameters(self):
        """ Returns the parameters as a flat array 
            [bh,w,q]
        """

        return np.concatenate([1.0/self._phase*self._bh.flatten(),1.0/self._phase*self._w.flatten(),self._q.diagonal()])

    def set_parameters(self, params):
        """ Set the matrices from flat input array P 
            P = [bh,w,q]
        """
        index = 0
        
        self._bh = self._phase*params[index:index+self._bh.shape[0]].reshape(self._bh.shape)
        index += self._bh.shape[0]

        self._w = self._phase*params[index:index+self._w.size].reshape(self._w.shape)
        index += self._w.size

        np.fill_diagonal(self._q, params[index:index+self._q.shape[0]])

    def get_bounds(self):
        """Returns two arrays with min and max of each parameter for the GA"""
        return [self._lower_bounds, self._upper_bounds]
    
    def get_gradients(self):
        """ Returns gradients as a flat array 
            [b,w,q]
        """
        return np.concatenate([self._gradB.flatten(),self._gradW.flatten(),self._gradQ.diagonal()])
    
    def backprop(self, E):
        """ Propagates the error E through the layer and stores gradient """
       
        # Calc ingredients
        vWb = np.transpose(self._X).dot(self._w)+self._bh.T
        
        T1n = np.zeros((self._Nout, self._X.shape[1]))
        T2n = np.zeros((self._Nout, self._X.shape[1]))
        T3n = np.zeros((self._Nout, self._X.shape[1]))
       
        if(self._phase!=1):
            for i in range(0,self._Nout):  
                O = np.matrix([[self._q[i, i]]], dtype=complex)
            
                T0 = theta_1d( vWb[:, [i]], O, 0)
               
                T1n[i] = theta_1d( vWb[:, [i]], O, 1)/T0
                T2n[i] = theta_1d( vWb[:, [i]], O, 2)/T0
                T3n[i] = theta_1d( vWb[:, [i]], O, 3)/T0
              
        else:
            for i in range(0,self._Nout):  
                O = np.matrix([[np.real(self._q[i, i])]], dtype=float)
                
                T0 = logtheta_1d_phaseI( np.real(vWb[:, [i]]), O, 0)
                T1n[i] = np.exp(logtheta_1d_phaseI( np.real(vWb[:, [i]]), O, 1)-T0)
                T2n[i] = np.exp(logtheta_1d_phaseI( np.real(vWb[:, [i]]), O, 2)-T0)
                T3n[i] = np.exp(logtheta_1d_phaseI( np.real(vWb[:, [i]]), O, 3)-T0)
                
               # tic = time.clock()
                """
                r0 = pool.apply_async(logtheta_1d_phaseI, [np.real(vWb[:, [i]]), O, 0])  
                r1 = pool.apply_async(logtheta_1d_phaseI, [np.real(vWb[:, [i]]), O, 1])  
                r2 = pool.apply_async(logtheta_1d_phaseI, [np.real(vWb[:, [i]]), O, 2])  
                r3 = pool.apply_async(logtheta_1d_phaseI, [np.real(vWb[:, [i]]), O, 3])  
          
                T0 = np.exp(r0.get(timeout=10))
                T1n[i] = np.exp(r1.get(timeout=10)-T0)
                T2n[i] = np.exp(r2.get(timeout=10)-T0)
                T3n[i] = np.exp(r3.get(timeout=10)-T0)
                """
              
               # toc = time.clock()
               # print("4x logtheta: "+str(1000*(toc-tic))+"ms"+" O: ", O, "sV: ",vWb[:, [i]].shape ,"maxV: ",np.max(np.abs(np.real(vWb[:, [i]]))) )
            
        if(np.isnan(T1n).any() or np.isnan(T2n).any() or np.isnan(T3n).any()   ):
            print("NaN detected in T1n")
            print("T1n:",T1n)
            print("T2n:",T2n)
            print("T3n:",T3n)
        
        T1nSquare = T1n*T1n
        
        kappa = -(T2n-T1nSquare)
        
        # B grad
        self._gradB = np.mean(kappa*E,axis=1,keepdims=True)
        
        # Q grad
        rho = (T3n - T1n*T2n)*E
        
        self._gradQ = np.diag(np.mean(rho, axis=1).flatten())
        
        # W grad 
        delta = kappa*E
        
        self._gradW = delta.dot(self._X.T).T/self._X.shape[1]
       
        """
        print("***")
        print("gQ:",self._gradQ)
        print("gQs:",self._gradQ.shape)
        print("Qs:",self._q.shape)
        
        print("gB:",self._gradB)
        print("gBs:",self._gradB.shape)
        print("Bs:",self._bh.shape)
        
        print("gW:",self._gradW)
        print("gWs:",self._gradW.shape)
        print("Ws:",self._w.shape)
        """
    
        return self._w.dot(delta)