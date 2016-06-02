import sys
import os
import numpy as np
import theano as T
import theano.tensor as TT
from sklearn.base import BaseEstimator
from .tespo import tespo

class BASE(BaseEstimator):

    def _init_para(self, X, y):
        '''
        '''
        F =  X.shape[1]
        C = len(np.unique(y))
        M = y.shape[1]

        w = np.random.uniform(-1,1,(M, F))
        d = np.ones((M, C-2))

        # marginal function parameter
        b = np.ones(M) *(1 - 0.5 * C)
        s = np.ones(M)

        p0 = {}
        p0['b'] = tespo.parameter(b)
        p0['w'] = tespo.parameter(w)
        p0['d'] = tespo.parameter(d)
        p0['s'] = tespo.parameter(s)

        return p0, [F,C,M]
    
    def _z(self, para, X):
        w = para['w'].value
        z = T.dot(w, X.T)
        return z

    def _cdf(self, para, X):
        '''
        '''
        z = self._z(para, X)
        b = para['b'].value
        d = para['d'].value
        s = para['s'].value

        b = b.dimshuffle(0, 'x')
        NU = TT.extra_ops.cumsum(
            TT.concatenate((b, TT.sqr(d)), axis=1),
            axis=1)

        NU = TT.concatenate(
            (-1e20 * TT.ones_like(b), NU, 1e20 * TT.ones_like(b)),
            axis=1)

        NU = NU.dimshuffle('x', 0, 1)
        Z = z.dimshuffle(1, 0, 'x')
        Z = TT.extra_ops.repeat(Z, NU.shape[2], 2)
        S = s.dimshuffle('x', 0, 'x')

        cdf = self._margin(NU, TT.sqr(S), Z)

        return cdf 

    def _pdf(self, para, X):
        cdf = self._cdf(para, X)
        pdf = cdf[:, :, :-1]-cdf[:, :, 1:]
        return pdf

    def score(self,X,y):
        y_hat = self.predict(X)
        return -np.mean((y-y_hat)**2,0).mean()
