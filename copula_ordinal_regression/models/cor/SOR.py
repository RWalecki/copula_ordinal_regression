import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__)+'/../', os.path.pardir)))

import tespo
import numpy as np
import theano as T
import theano.tensor as TT

from statistics import *
from marginals import sigmoid, normcdf
from sklearn.base import BaseEstimator




class SOR(BaseEstimator):

    hyper_parameters_quick = {
            'C':[0]+10.**np.arange(0,5),
            'margins':['normcdf','sigmoid'],
            }

    hyper_parameters = {
            'C':[0]+10.**np.arange(0,10),
            'margins':['normcdf','sigmoid'],
            'loss_function':['mse','ncll'],
            'output':['MAP','expectation'],
            }

    def __init__(
            self,
            C = 1e5,
            margins = 'sigmoid',
            loss_function = 'ncll',
            output = 'MAP',
            verbose = 0,
            max_iter = 15000,
            ):
        """
        """
        self.C = C
        self.max_iter = max_iter
        self.verbose = verbose
        self.margins = margins
        self.loss_function = loss_function
        self.output = output

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

        if self.margins=='sigmoid':self._margin = sigmoid
        if self.margins=='normcdf':self._margin = normcdf

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

    def _predict(self, para, X):
        P = self._pdf(para, X)

        if self.output=='expectation':
            return expectation(P)
        if self.output=='MAP':
            return TT.argmax(P, 2)
        if self.output=='probability':
            return P

    def _loss(self, para, X, y):
        '''
        '''
        P = self._pdf(para, X)

        if self.loss_function=='mse':
            E = expectation(P)
            loss = TT.mean( TT.sqr(E-y) )

        if self.loss_function=='ncll':
            NCLL = node_potn(P, y)
            loss = TT.mean( NCLL )

        if self.C:
            loss += 1./float(self.C) * TT.sum(TT.sqr(para['w'].value))

        return loss

    def fit(self, X, y, debug=0):
        self.p0, self.shape = self._init_para(X, y)
        self.p1 = self.p0

        # dry-run
        if debug==True:
            tespo.debug(self._pdf,  [self.p0, X])
            tespo.debug(self._loss, [self.p0, X, y])
            tespo.debug(self._predict, [self.p0, X])
        """
        """
        # if self.weights==None:self.w = 1/X.shape[0]
        # self.w = utils.statistics.weights(y,self.weights)
        self._predict_C = tespo.compile(self._predict, [self.p0, X], jac=False)
        self._loss_C, self._grad_C = tespo.compile(self._loss, [self.p0, X, y], jac=True)

        if self.verbose == 0:callback = None
        if self.verbose >= 1:callback = 'default'

        # start optimization
        self.p1, self.cost = tespo.optimize(
            fun=self._loss_C,
            p0=self.p0,
            jac=self._grad_C,
            callback=callback,
            method='CG',
            args=(X, y),
            options = {'maxiter': self.max_iter, 'disp': self.verbose > 1}
        )
        return self

    def predict(self, X):
        para = tespo.utils.para_2_vector(self.p1)
        y_hat = self._predict_C(para, X)
        return y_hat

    def score(self,X,y):
        y_hat = self.predict(X)
        return -np.mean((y-y_hat)**2,0).mean()
