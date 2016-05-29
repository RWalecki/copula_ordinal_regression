import cPickle
import gzip
import numpy as np
import copula_ordinal_regression as cor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict 
from sklearn.model_selection import LabelKFold , KFold


dat = cPickle.load(gzip.open('./tests/data/disfa_slim.pklz','rb'))
X = np.vstack(dat['X'])
y = np.vstack(dat['y'])
S = np.hstack([[ii]*jj.shape[0] for ii,jj in zip(dat['S'],dat['y'])])


cv = LabelKFold(9)

clf = cor.models.MLR(max_iter=500,verbose=0)

clf = GridSearchCV(
        clf,
        clf.hyper_parameters,
        cv = cv,
        n_jobs= -1,
        verbose = 10,
        )
clf.fit(X,y,S)

# import ipdb; ipdb.set_trace()

y_hat = cross_val_predict(
        clf.best_estimator_,
        X, y, S,
        n_jobs=1,
        cv = cv,
        verbose = 10
        )


print np.mean((y-y_hat)**2,0).mean()
print cor.utils.metrics.ICC(y,y_hat)
print cor.utils.metrics.ICC(y,y_hat).mean()
print clf.best_params_
