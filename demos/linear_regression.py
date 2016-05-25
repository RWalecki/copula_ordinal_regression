import cPickle
import gzip
import numpy as np
import copula_ordinal_regression as cor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict 
from sklearn.model_selection import LabelKFold , KFold
import mula_learn as mll


dat = cPickle.load(gzip.open('./tests/data/disfa_slim.pklz','rb'))
X = np.vstack(dat['X'])
y = np.vstack(dat['y'])
S = np.hstack([[ii]*jj.shape[0] for ii,jj in zip(dat['S'],dat['y'])])

cv = KFold(2)

clf = cor.models.MLR(max_iter=1000)

clf = GridSearchCV(
        clf,
        clf.hyper_parameters,
        cv = cv,
        n_jobs= -1,
        verbose = 10,
        )
clf.fit(X,y)

y_hat = cross_val_predict(
        clf.best_estimator_,
        X, y, 
        cv = cv,
        verbose = 10
        )


print np.mean((y-y_hat)**2,0).mean()
print mll.utils.metrics.ICC(y,y_hat)
print mll.utils.metrics.ICC(y,y_hat).mean()
