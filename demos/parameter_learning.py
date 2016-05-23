import copula_ordinal_regression.tespo as tespo
import theano as T
import theano.tensor as TT
import numpy as np


######################################################
# generate random dataset:
N = 600 # samples
F = 5   # features
C = 4   # labels
y = np.random.randint(0, C, N)
X = np.random.rand(N, F) * np.array([y+1]).T
######################################################

# initial model 
w = np.random.uniform(-1,1,(F, C))
b = np.zeros(C)
p0 = {}
p0['w'] = tespo.parameter(w, const=False)
p0['b'] = tespo.parameter(b, const=False)


# define loss function
def loss(pi, X, y):
    '''
    compute loss (average negative conditional log likelihood)
    '''
    w = pi['w'].value
    b = pi['b'].value
    P = TT.nnet.softmax( T.dot(X, w) + b )
    idx = TT.arange(y.shape[0]).astype('int64')
    idy = y.astype('int64')
    return -TT.mean(TT.log(P[idx, idy]))


# define prediction function
def pred(pi, X):
    '''
    compute predictions
    '''
    w = pi['w'].value
    b = pi['b'].value
    P = TT.nnet.softmax( T.dot(X, w) + b )
    return TT.argmax(P, 1)


# debug your theano code (optional)
tespo.debug(pred, [p0, X])
tespo.debug(loss, [p0, X, y])


# compile your functions
pred_T = tespo.compile(pred, [p0, X], jac=False)
loss_T, grad_T = tespo.compile(loss, [p0, X, y], jac=True)


# # exection of comiled funtion (optional)
tespo.exe(pred_T, [p0, X])
tespo.exe(loss_T, [p0, X, y])


# define your own callback function (optional)
def callback(pi):
    Y_hat = tespo.exe(pred_T, [pi, X])
    out = {}
    out['ACC']  = np.mean(Y_hat==y)
    out['Loss'] = tespo.exe(loss_T, [pi, X,y])
    out['pi']   = 3.14
    opt = {'freq':1}
    return out, opt


# use here the features and options from scipy.optimize 
# to learn model parameter
p1, res = tespo.optimize(
    fun=loss_T,
    p0=p0,
    jac=grad_T,
    callback=callback,
    args=(X, y),
    method='BFGS',
    options = {'maxiter': 20, 'disp': 0},
)


# compute predictions
Y_hat = tespo.exe(pred_T, [p1, X])
loss  = tespo.exe(loss_T, [p1, X, y])
