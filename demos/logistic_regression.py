import cPickle
import gzip
import numpy as np
import copula_ordinal_regression as cor

dat = cPickle.load(gzip.open('./tests/data/disfa_slim.pklz','rb'))
X = np.vstack(dat['X'])
y = np.vstack(dat['y'])
print X.shape
print y.shape

clf = cor.models.mlr(X, y, verbose=1, C=0, max_iter=200)
clf.fit(X,y)
y_hat = clf.predict(X)

print np.mean((y-np.zeros_like(y))**2,0).mean()
print np.mean((y-y_hat)**2,0).mean()
print np.mean((y-np.random.randint(0,5,y.shape))**2,0).mean()
