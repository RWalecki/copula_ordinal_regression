import copula_ordinal_regression.tespo as tespo 
import numpy as np
import theano as T
import theano.tensor as TT

def rand_dset(N=1000,F=12,C=4):
    '''
    generate random dataset:
    N samples
    F features
    C discrete labels
    '''
    y = np.random.randint(0, C, N)
    X = np.random.rand(N, F)*np.array([y+1]).T
    return X,y

def init(X,y,r=0):
    '''
    '''
    N, F = X.shape
    C = len(np.unique(y))

    w = np.random.rand(F, C)
    b = np.ones(C)

    para = {}
    para['w'] = tespo.parameter(w, const=False)
    para['b'] = tespo.parameter(b, const=False)
    para['r'] = tespo.parameter(r, const=True)

    return para

def loss(para, X, y):
    '''
    '''
    w = para['w'].value
    b = para['b'].value
    r = para['r'].value
    P = TT.nnet.softmax( T.dot(X, w) + b )
    idx = TT.arange(y.shape[0]).astype('int16')
    idy = y.astype('int16')
    loss = -TT.mean(TT.log(P[idx, idy]))

    if r:
        return loss + 1/r * TT.sqr(w)
    else:
        return loss

def pred(p, X):
    '''
    '''
    w = p['w'].value
    b = p['b'].value
    P = TT.nnet.softmax( T.dot(X, w) + b )
    return TT.argmax(P, 1)
