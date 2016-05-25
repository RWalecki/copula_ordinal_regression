import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__)+'/../', os.path.pardir)))

import tespo
import numpy as np
import theano as T
import theano.tensor as TT
from sklearn.base import BaseEstimator, ClassifierMixin


def sigmoid(nu, sigma, z):
    return TT.nnet.sigmoid((z-nu)/sigma)

def normcdf(nu, sigma, z):
    return 0.5 * ( 1 + TT.erf( (z-nu) / (sigma * (2**0.5) ) ) )

def weights(y,type):
    if type==None:
        return np.ones_like(y)/np.float64(y.shape[0])
    if type=='balanced':
        lb = preprocessing.LabelBinarizer()
        lb.fit(y.flatten())
        res = np.zeros_like(y)
        for i in range(y.shape[1]):
            w_ = np.sum(lb.transform(y[:,i]),0)
            res[:,i] = w_[y[:,i]]
        return (np.float64(res)**-1.)

def expectation(P):
    states = TT.arange(P.shape[2]).dimshuffle('x','x',0)
    states = TT.extra_ops.repeat(states,P.shape[0],0)
    states = TT.extra_ops.repeat(states,P.shape[1],1)
    return TT.sum(P*states,axis=2)

def log_prob(P, realmin = 1e-20):
    '''
    numerical stabil log function
    for some reason theano needs a high realmin value
    '''

    # if prob is less than realmin, set it to realmin
    idx = TT.le(P,realmin).nonzero()
    P = TT.set_subtensor(P[idx],realmin)

    # if prob larger than 1-realmin, set it to 1-realmin
    idx = TT.ge(P,1-realmin).nonzero()
    P = TT.set_subtensor(P[idx],1-realmin)

    idx = TT.isnan(P).nonzero()
    P = TT.set_subtensor(P[idx],realmin)

    idx = TT.isinf(P).nonzero()
    P = TT.set_subtensor(P[idx],realmin)

    return TT.log(P)

def compute_cll(pdf, y=None):
    '''
    pdf:     [AU X Frame X Label]

    node that there are M+1 Thresholds and they have to go from 0 to 1
    '''
    if y:
        y_ = y.T.astype('int8').flatten(1)
        pdf = pdf.T.flatten(2)
        idx = TT.arange(y_.shape[0])
        P = pdf[y_,idx]
        P = P.reshape(y.shape)
    else:
        P = pdf

    return log_prob(P)

class SOR(BaseEstimator):

    def __init__(
            self,
            C=0,
            margins = 'normcdf',
            loss_function = 'ncll',
            output = 'MAP',
            verbose = 0,
            max_iter = 500,
            ):
        """
        """
        self.C = C
        self.max_iter = max_iter
        self.verbose = verbose
        self.margins = margins
        self.loss_function = loss_function
        self.output = output

        self.hyper_parameters = {
                'C':10.**np.arange(-4,5),
                'margins':['normcdf','sigmoid'],
                'loss_function':['mse','ncll'],
                'output':['MAP','expectation'],
                }

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

        para = {}
        para['b'] = tespo.parameter(b)
        para['w'] = tespo.parameter(w)
        para['d'] = tespo.parameter(d)
        para['s'] = tespo.parameter(s)

        return para, [F,C,M]
    
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
            CLL = compute_cll(P, y)
            loss = -TT.mean( CLL )

        if self.C:
            loss += 1./float(self.C) * TT.sum(TT.sqr(para['w'].value))

        return loss

    def fit(self, X, y, debug=0):
        self.p0, self.shape = self._init_para(X, y)
        self.p1 = self.p0

        if self.margins=='sigmoid':self._margin = sigmoid
        if self.margins=='normcdf':self._margin = normcdf

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
        para = tespo.utils.para_2_vector(self.p1)
        y_hat = self._predict_C(para, X)
        return np.mean((y-y_hat)**2,0).mean()
