import sys
import os
import numpy as np
import theano as T
import theano.tensor as TT
import itertools
from pystruct.inference import inference_ad3
from sklearn.covariance import GraphLasso

from .tespo import tespo
from .statistics import node_potn, edge_potn
from .copulas import frank, indep, gumbel
from .marginals import sigmoid, normcdf

from .BASE import BASE

class COR(BASE):

    hyper_parameters_quick = {
            'C':[1e6],
            'w_nodes':np.linspace(0,1,10),
            'shared_copula':[False,True],
            }

    hyper_parameters = {
            'C':[0]+10.**np.arange(0,10),
            'margins':['normcdf','sigmoid'],
            'shared_copula':[False,True],
            'w_nodes':np.linspace(0,1,10),
            }

    def __init__(
            self,
            C = 0,
            margins = 'sigmoid',
            copula = 'frank',
            optimizer = 'CG',
            sparsity = 0,
            w_nodes = 0.1,
            shared_copula = True,
            verbose = 0,
            max_iter = 5000,
            ):
        self.C = C
        self.max_iter = max_iter
        self.optimizer = optimizer
        self.verbose = verbose
        self.margins = margins
        self.sparsity = sparsity
        self.w_nodes = w_nodes
        self.copula = copula
        self.shared_copula = shared_copula 

    def _init_para(self, X, y):
        '''
        '''
        p0, shape = BASE._init_para(self, X, y)

        edges = []
        if self.sparsity==0:
            for i in itertools.combinations(range(y.shape[1]), 2):
                e1 = min(i)
                e2 = max(i)
                edges.append([e1, e2])
        else:
            lasso = GraphLasso(alpha = 0.1)
            lasso.fit(y)
            graph = lasso.get_precision()!=0
            for i in range(graph.shape[0]):
                for j in range(i+1,graph.shape[1]):
                    if graph[i,j]==0:continue
                    edges.append([i, j])

        self.edges = T.shared(np.array(edges).T).astype('int8')

        if self.verbose:
            print('(edges)',edges)
            print('(y) shape:',y.shape,'labels:',np.unique(y))
            print('(X) shape:',X.shape,'std:',np.std(X))



        theta = []
        for e1, e2 in edges:
            if self.shared_copula:
                theta.append(0.01)
            else:
                theta.append(np.ones((shape[1],shape[1]))*0.01)

        p0['theta'] = tespo.parameter(theta, const=False)
        return p0, shape

    def _loss(self, para, X, y):
        '''
        '''
        w_nodes = self.w_nodes
        w_edges = 1-self.w_nodes

        theta = para['theta'].value
        edges = self.edges

        P = self._pdf(para, X)

        loss = T.shared(0)
        if w_nodes:
            NCLL = node_potn(P, y)
            loss += TT.mean( NCLL ) * self.w_nodes

        if w_edges:
            NCJLL = edge_potn(P,self._copula,theta,edges,y,shared_copula=self.shared_copula)
            loss += TT.mean( NCJLL ) * (1-self.w_nodes)

        if self.C:
            loss += 1./self.C * TT.sum(TT.sqr(para['w'].value))

        return loss

    def predict(self, X,y=None,w=1):
        p1 = self.p1
        for i in p1:p1[i].value = T.shared(p1[i].value)

        edges = self.edges

        pdf = self._pdf(p1, X)
        theta = p1['theta'].value

        X = node_potn(pdf).eval()*self.w_nodes
        F = edge_potn(pdf, self._copula, theta, edges,shared_copula=self.shared_copula).dimshuffle(1,0,2,3).eval()*(1-self.w_nodes)
        E = edges.eval()

        y_hat = np.array([inference_ad3(-x,-f,E.T) for x,f in zip(X,F)])

        return y_hat

    def fit(self, X, y, debug=0):
        self.p0, self.shape = self._init_para(X, y)
        self.p1 = self.p0

        if self.margins=='sigmoid':self._margin = sigmoid
        if self.margins=='normcdf':self._margin = normcdf

        if self.copula=='frank':self._copula = frank
        if self.copula=='indep':self._copula = indep 
        if self.copula=='gumbel':self._copula = gumbel

        if debug==True:
            tespo.debug(self._loss, [self.p0, X, y])
            return self

        # compile cost objective and gradients
        self._loss_C, self._grad_C = tespo.compile(self._loss, [self.p0, X, y], jac=True)


        def callback(pi):
            out = {}
            out['Loss']  = tespo.exe(self._loss_C, [pi, X,y])
            opt = {'freq':1}
            return out, opt

        if self.verbose == 0:callback = None

        # start optimization
        self.p1, self.cost = tespo.optimize(
            fun=self._loss_C,
            p0=self.p0,
            jac=self._grad_C,
            callback=callback,
            method=self.optimizer,
            args=(X, y),
            options = {'maxiter': self.max_iter, 'disp': self.verbose > 1}
        )
        return self
