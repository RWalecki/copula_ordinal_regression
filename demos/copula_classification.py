import cPickle
import gzip
import numpy as np
import copula_ordinal_regression as cor
from sklearn import grid_search

dat = cPickle.load(gzip.open('./tests/data/disfa_slim.pklz','rb'))
X = np.vstack(dat['X'])
y = np.vstack(dat['y'])
print X.shape
print y.shape

clf = cor.models.sor(max_iter=10)
# clf.fit(X,y)
# y_hat = clf.predict(X)

# print np.mean((y-np.zeros_like(y_hat))**2,0).mean()
# print np.mean((y-y_hat)**2,0).mean()

clf = grid_search.GridSearchCV(
        clf,
        clf.hyper_parameters,
        n_jobs=-1
        )
clf.fit(X,y)

print np.array([exp[1] for exp in clf.grid_scores_])
