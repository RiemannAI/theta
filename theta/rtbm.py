# -*- coding: utf-8 -*-
from __future__ import absolute_import
import numpy as np
from theta.mathtools import rtbm_probability, hidden_expectations, rtbm_log_probability, \
    check_normalization_consistency, check_pos_def, factorized_rtbm_probability

from theta.riemann_theta.riemann_theta import RiemannTheta


class AssignError(Exception):
    pass


class RTBM(object):
    """This class implements the Riemann Theta Boltzmann Machine"""
    class Mode:
        Probability = 0
        LogProbability = 1
        Expectation = 2

    def __init__(self, visible_units, hidden_units, mode=Mode.Probability,
                 init_max_param_bound=2, random_bound=1, phase=1, diagonal_T=False, diagonal_Q=False):
        """Setup operators for BM based on the number of visible and hidden units

        Args:
            visible_units: number of visible units
            hidden_units: number of hidden units
            mode: see Mode enumerator
            init_max_param_bound: size of maximum parameters used in random initialization.
            random_bound: selects the maximum random value for the Schur complement initialization
            phase: number which multiplies w and bh
            diagonal_T: force T diagonal, by default T is symmetric
            check_positivity: verifies if set_parameters satisfy positivity condition

        """
        self._Nv = visible_units
        self._Nh = hidden_units
        self._bv = np.zeros([visible_units, 1])
        self._t = np.ones([visible_units, visible_units])
        self._bh = np.zeros([hidden_units, 1])
        self._w = np.zeros([visible_units, hidden_units])
        self._q = np.zeros([hidden_units, hidden_units])
        self._diagonal_T = diagonal_T
        self._diagonal_Q = diagonal_Q
        self._mode = None
        self._call = None
        self._parameters = None
        self._X = None
        self._phase = phase
        self._check_positivity = True

        self._size =  self._Nv + self._Nh + self._Nv*self._Nh
        if diagonal_T and diagonal_Q:
            self._size += self._Nv + self._Nh
        elif diagonal_T and not diagonal_Q:
            self._size += self._Nv + (self._Nh**2+self._Nh)//2
        elif not diagonal_T and diagonal_Q:
            self._size += self._Nh + (self._Nv**2+self._Nv)//2
        else:
            self._size = self._Nv + self._Nh + (self._Nv**2+self._Nv+self._Nh**2+self._Nh)//2+self._Nv*self._Nh

        # set operation mode
        self.mode = mode

        # set boundaries
        self.set_bounds(init_max_param_bound)
        
        # Populate with random parameters using Schur complement
        # This guarantees an acceptable and instantaneous initial solution
        self.random_init(random_bound)
        
        # Generate vector for gradient calc call
        self._D1 = []
        for i in range(hidden_units):
            tmp = [0] * hidden_units
            tmp[i] = 1
            self._D1.append(tmp)

        self._D1 = np.array(self._D1)

        # Generate vector for hessian calc call
        self._D2 = []
        
        if(hidden_units > 1):
            for i in range(0, hidden_units):
                for j in range(0, hidden_units):
                    tmp = [0] * hidden_units**2
                    tmp[i] = 1
                    tmp[j+hidden_units] = 1
            
                    self._D2.append(tmp)

            self._D2 = np.array(self._D2) 
        else:
            self._D2.append([1,1])
    
    def __call__(self, data, grad_calc=False):
        """Evaluates the RTBM instance for a given data array"""
        
        P = self._call(data)
        # Store for backprop
       
        if grad_calc:
            self._X = data
            self._P = P
            self._check_positivity = False
        else:
            self._check_positivity = True
            
        return P

    def feed_through(self, X, grad_calc=False):
        return self.__call__(X, grad_calc=grad_calc)

    def size(self):
        """Get size of RTBM"""
        return self._size

    def random_init(self, bound):
        """A fast random initializer based on Schur complement, if diagonal_T=True
        the initial Schur complement is defined diagonal, so Q and T are diagonal and W is zero.

        Args:
            bound: the maximum value for the random matrix X used by the Schur complement
        """
        a_shape = ((self._Nv+self._Nh), (self._Nv+self._Nh))
        a_size = (self._Nv+self._Nh)**2

        params = np.random.uniform(-bound, bound, a_size+self._Nv+self._Nh)
        if self._diagonal_T or self._diagonal_Q:
            x = np.eye(a_shape[0])
            np.fill_diagonal(x, params[:self._Nv+self._Nh])
        else:
            x = params[:a_size].reshape(a_shape)

        a = np.transpose(x).dot(x)

        self._q = a[:self._Nh, :self._Nh]
        self._t = a[self._Nh:self._Nh + self._Nv, self._Nh:]
        self._w = self._phase * a[self._Nh:, :self._Nh]
        self._bv = params[a_size:a_size + self._Nv].reshape(self._bv.shape)
        self._bh = self._phase * params[-self._Nh:].reshape(self._bh.shape)

        # store parameters having in mind that Q and T are symmetric.
        if self._diagonal_T and not self._diagonal_Q:
            self._parameters = np.concatenate([self._bv.flatten(), self._bh.flatten(),
                                               self._w.flatten(), self._t.diagonal(),
                                               self._q[np.triu_indices(self._Nh)]])
        elif not self._diagonal_T and self._diagonal_Q:
            self._parameters = np.concatenate([self._bv.flatten(), self._bh.flatten(),
                                               self._w.flatten(), self._t[np.triu_indices(self._Nv)],
                                               self._q.diagonal()])
        elif self._diagonal_T and self._diagonal_Q:
            self._parameters = np.concatenate([self._bv.flatten(), self._bh.flatten(),
                                               self._w.flatten(), self._t.diagonal(),
                                               self._q.diagonal()])
        else:
            self._parameters = np.concatenate([self._bv.flatten(), self._bh.flatten(),
                                               self._w.flatten(), self._t[np.triu_indices(self._Nv)],
                                               self._q[np.triu_indices(self._Nh)]])

    def set_parameters(self, params):
        """Assigns a flat array of parameters to the RTBM matrices.

        Args
            params: list of parameters to populate Bv, Bh, W, T, Q

        Returns:
            True if Q, T and Q-WtTW are positive, False otherwise.
        """

        if len(params) != self._size:
            raise Exception('Size does no match.')

        self._parameters = params

        self._bv = params[:self._Nv].reshape(self._bv.shape)
        index = self._Nv

        self._bh = self._phase*params[index:index+self._Nh].reshape(self._bh.shape)
        index += self._Nh

        self._w = self._phase*params[index:index+self._Nv*self._Nh].reshape(self._w.shape)
        index += self._w.size

        if self._diagonal_T:
            np.fill_diagonal(self._t, params[index:index+self._Nv])
            index += self._Nv
        else:
            inds = np.triu_indices_from(self._t)
            self._t[inds] = params[index:index+(self._Nv**2+self._Nv)//2]
            self._t[(inds[1], inds[0])] = params[index:index+(self._Nv**2+self._Nv)//2]
            index += (self._Nv**2+self._Nv)//2

        if self._diagonal_Q:
            np.fill_diagonal(self._q, params[index:index+self._Nh])
            index += self._Nh
        else:
            inds = np.triu_indices_from(self._q)
            self._q[inds] = params[index:index+(self._Nh**2+self._Nh)//2]
            self._q[(inds[1], inds[0])] = params[index:index+(self._Nh**2+self._Nh)//2]

        if self._check_positivity:
            if not check_normalization_consistency(self._t, self._q, self._w) or \
                    not check_pos_def(self._q) or not check_pos_def(self._t):
                return False
        return True

    def get_parameters(self):
        """Return flat array with current matrices weights """
        return self._parameters

    def get_gradients(self):
        """Return flat array with calculated gradients 
           [Gbh,Gbv,Gw,Gt,Gq]
        """
        inds = np.triu_indices_from(self._gradQ)
        
        if self._diagonal_T and not self._diagonal_Q:
            return np.real(np.concatenate((self._gradBv.flatten(),self._gradBh.flatten(),self._gradW.flatten(), self._gradT.diagonal(), self._gradQ[inds].flatten()  )))
        elif not self._diagonal_T and self._diagonal_Q:
            return np.real(np.concatenate((self._gradBv.flatten(),self._gradBh.flatten(),self._gradW.flatten(), self._gradT.flatten(), self._gradQ.diagonal()  )))
        elif self._diagonal_T and self._diagonal_Q:
            return np.real(np.concatenate((self._gradBv.flatten(),self._gradBh.flatten(),self._gradW.flatten(), self._gradT.diagonal(), self._gradQ.diagonal()  )))
        else:
            return np.real(np.concatenate((self._gradBv.flatten(),self._gradBh.flatten(),self._gradW.flatten(), self._gradT.flatten(), self._gradQ[inds].flatten()  )))

    def get_bounds(self):
        """Returns two arrays with min and max of each parameter for the GA"""
        return self._bounds

    def set_bounds(self, param_bound):
        upper_bounds = [param_bound] * self._size
        lower_bounds = [-param_bound] * self._size
        self._bounds = [lower_bounds, upper_bounds]

    def mean(self):
        """compute mean estimator"""
        if self._mode is self.Mode.Probability:
            invT = np.linalg.inv(self._t)
            BvT = self._bv.T
            BhT = self._bh.T
            BtiTW = np.dot(np.dot(BvT, invT), self._w)
            WtiTW = np.dot(np.dot(self._w.T, invT), self._w)
            return np.real(1.0 / (2j * np.pi) * invT * self._w *
                           RiemannTheta.normalized_eval((BhT - BtiTW) / (2.0j * np.pi), (-self._q + WtiTW) / (2.0j * np.pi),
                                                        mode=self._mode, derivs=np.array([[1]])))
        else:
            assert AssertionError('Mean for mode %s not implemented' % self._mode)
    
    @property
    def mode(self):
        return self._mode

    @mode.setter
    def mode(self, value):

        mode = 0
        if self._phase == 1:
            mode = 1
        elif self._phase == 1j:
            mode = 2

        if value is self.Mode.Probability:
            if self._diagonal_Q:
                self._call = lambda data: np.real(factorized_rtbm_probability(data, self._bv, self._bh, self._t, self._w, self._q, mode))
            else:
                self._call = lambda data: np.real(rtbm_probability(data, self._bv, self._bh, self._t, self._w, self._q, mode))
        elif value is self.Mode.LogProbability:
            self._call = lambda data: np.real(rtbm_log_probability(data, self._bv, self._bh, self._t, self._w, self._q, mode))
        elif value is self.Mode.Expectation:
            self._call = lambda data: np.real(self._phase*hidden_expectations(data, self._bh, self._w, self._q))
        else:
            raise AssertionError('Mode %s not implemented.' % value)

        self._mode = value


    def backprop(self, E):
        
        if self._diagonal_T:
            
            vWb = np.transpose(self._X).dot(self._w)+self._bh.T
            iT  = 1.0/self._t
            iTW = iT.dot(self._w)

            # Gradients
            arg1 = vWb / (2.0j * np.pi)
            arg2 = -self._q/ (2.0j * np.pi)
            arg3 = (self._bh.T-self._bv.T.dot(iTW)) / (2.0j * np.pi)
            arg4 = -(self._q-self._w.T.dot(iTW))/ (2.0j * np.pi)
            coeff1 = 1.0/(2.0j*np.pi)
            coeff2 = np.square(coeff1)

            Da = coeff1*RiemannTheta.normalized_eval(arg1, arg2, mode=1, derivs=self._D1)
            
            Db = coeff1*RiemannTheta.normalized_eval(arg3, arg4, mode=1, derivs=self._D1)
            
            # Hessians
            DDa = coeff2*RiemannTheta.normalized_eval(arg1, arg2, mode=1, derivs=self._D2)
            
            DDb = coeff2*RiemannTheta.normalized_eval(arg3, arg4, mode=1, derivs=self._D2)
            
            # H from DDb
            Hb = DDb.flatten().reshape(self._q.shape)
            np.fill_diagonal(Hb, Hb.diagonal()*0.5)
            
            # Grad Bv
            self._gradBv = np.mean(E*(self._P*( -self._X -2.0*iT.dot(self._bv) + iTW.dot(Db))), axis=1)
            
            # Grad Bh
            self._gradBh = np.mean(E*self._P*(Da-Db), axis=1) 
           
            # Grad W
            self._gradW = (E*self._P*self._X).dot(Da.T)/self._X.shape[1] + np.mean(E*self._P, axis=1)*(  self._bv.T.dot(iT).T.dot(Db.T)  - 2*iTW.dot(Hb))
   
            # Grad T
            iT2 = np.square(iT)
            
            self._gradT = np.diag(np.mean(-0.5*self._P*self._X**2*E, axis=1)) + np.mean(E*self._P, axis=1)*(0.5*iT + self._bv**2*iT2 -self._bv*iT2*self._w.dot(Db) + iT2*self._w.dot(Hb).dot(self._w.T) )   
            
            # Grad Q
            self._gradQ = np.mean(-self._P*( DDa - DDb )*E, axis=1).reshape(self._q.shape)
            np.fill_diagonal(self._gradQ, self._gradQ.diagonal()*0.5)
        else:
            raise AssertionError('Gradients for non-diagonal T not implemented.')
            
    @property
    def bv(self):
        return self._bv

    @bv.setter
    def bv(self, value):
        if value.shape != self._bv.shape:
            raise AssertionError('Setting bv with wrong shape.')
        self._bv = value

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, value):
        if value.shape != self._t.shape:
            raise AssertionError('Setting t with wrong shape.')
        self._t = value

    @property
    def bh(self):
        return self._bh

    @bh.setter
    def bh(self, value):
        if value.shape != self._bh.shape:
            raise AssertionError('Setting bh with wrong shape.')
        self._bh = value

    @property
    def w(self):
        return self._w

    @w.setter
    def w(self, value):
        if value.shape != self._w.shape:
            raise AssertionError('Setting w with wrong shape.')
        self._w = value

    @property
    def q(self):
        return self._q

    @q.setter
    def q(self, value):
        if value.shape != self._q.shape:
            raise AssertionError('Setting q with wrong shape.')
        self._q = value
