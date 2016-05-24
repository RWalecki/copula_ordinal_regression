mport sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__)+'/../', os.path.pardir)))
import numpy as np
import theano as T
import theano.tensor as TT
import tespo
import cPickle,gzip


class mlr():

    def __init__(
            self,
            X,
            y,
            C=1e3,
            max_iter=100,
            verbose=0,
    ):
        """
        """
        self.C = C
        self.max_iter = max_iter
        self.verbose = verbose
        self.p0 = self._init_para(X, y)
        self.Output = np.unique(y)
        self.p1 = self.p0
        self.C_default = np.power(10.,np.arange(-5,6))
    
        # dry-run
        tespo.debug(self._loss, [self.p0, X, y])
        tespo.debug(self._predict, [self.p0, X])

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
        r = TT.dot(X,w.T)+b
        loss = TT.mean(TT.sqr(y-r))
        if self.C:
            loss += 1./self.C * TT.sum(TT.sqr(para['w'].value))
        return loss

    def fit(self, X, y):
        """
        """

        self.predict_C = tespo.compile(
            self._predict, [self.p0, X], jac=False)

        self.loss_C, self.loss_grad_C = tespo.compile(
            self._loss, [self.p0, X, y], jac=True)

        if self.verbose == 0:callback = None
        if self.verbose >= 1:callback = 'default'

        # start optimization
        self.p1, self.cost = tespo.optimize(
            fun=self.loss_C,
            p0=self.p0,
            jac=self.loss_grad_C,
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
        y_hat = self.predict_C(para, X)
        return y_hat
