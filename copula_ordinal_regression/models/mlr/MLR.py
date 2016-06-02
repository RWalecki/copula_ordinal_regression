import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__)+'/../', os.path.pardir)))
import numpy as np
import theano as T
import theano.tensor as TT
import tespo
import cPickle,gzip
from sklearn.base import BaseEstimator

class MLR(BaseEstimator):

    hyper_parameters = {
            'C':[0]+10.**np.arange(-3,7),
            }

    def __init__(
            self,
            C=1e10,
            max_iter=100,
            verbose=0,
    ):
        """
        """
        self.max_iter = max_iter
        self.verbose = verbose
        self.C = C
    

    def _init_para(self, X, y):
        '''
        '''
        np.random.seed(1)
        F, C, M = X.shape[1], len(np.unique(y)), y.shape[1]

        w = np.random.rand(M, F)-0.5
        b = np.zeros(M)

        para = {}
        para['b'] = tespo.parameter(b)
        para['w'] = tespo.parameter(w)

        return para

    def _predict(self, para, X):
        """
        """
        w = para['w'].value
        b = para['b'].value
        r = TT.dot(X,w.T)+b
        return r

    def _loss(self, para, X, y):
        w = para['w'].value
        b = para['b'].value
        y_hat = TT.dot(X,w.T)+b
        loss = TT.mean((y-y_hat)**2)
        if self.C:
            loss += 1/float(self.C) * TT.sum(TT.sqr(w))
        return loss

    def fit(self, X, y, debug=0):
        """
        """
        self.p0 = self._init_para(X, y)

        if debug:
            tespo.debug(self._loss, [self.p0, X, y])
            tespo.debug(self._predict, [self.p0, X])
            return self

        # compile theano functions
        self._predict_C = tespo.compile(
            self._predict, [self.p0, X], jac=False)
        self._loss_C, self._loss_grad_C = tespo.compile(
            self._loss, [self.p0, X, y], jac=True)

        if self.verbose == 0:callback = None
        if self.verbose >= 1:callback = 'default'

        # start optimization
        self.p1, self.cost = tespo.optimize(
            fun=self._loss_C,
            p0=self.p0,
            jac=self._loss_grad_C,
            callback=callback,
            method='CG',
            args=(X, y),
            options = {'maxiter': self.max_iter, 'disp': self.verbose > 1}
        )
        return self

    def predict(self, X):
        """
        """
        para = tespo.utils.para_2_vector(self.p1)
        y_hat = self._predict_C(para, X)
        return y_hat

    def score(self,X,y):

        para = tespo.utils.para_2_vector(self.p1)
        y_hat = self._predict_C(para, X)
        return -np.mean((y-y_hat)**2,0).mean()
