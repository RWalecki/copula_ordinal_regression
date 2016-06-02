from sklearn.model_selection import LabelKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_predict 
import copula_ordinal_regression as cor
import numpy as np

# load the processed disfa database
X, y, S  = cor.load_disfa()

# select the first 3 action units (AU1,AU2,AU4)
y = y[:,[0,1,2]]


# select estimator and number of folds for cross validation
clf = cor.COR(max_iter=5000, verbose=0)
cv = LabelKFold(9)

# define parameter grid
parameter = {
        'margins':['normcdf','sigmoid'],
        'C':[0]+10.**np.arange(0,8),
        'w_nodes':np.linspace(0,1,5),
        }

# apply grid search to find optimal hyper parameters
clf = GridSearchCV(
        clf,
        parameter,
        cv = cv,
        n_jobs= -1,
        verbose = 10,
        refit=False
        )
clf.fit(X,y,S)
print clf.best_params_

# apply cross validation using best hyper parameters
y_hat = cross_val_predict(
        clf.best_estimator_,
        clf,
        X, y, S,
        n_jobs=1,
        cv = cv,
        verbose = 10
        )


# print resutls on test set
print cor.metrics.ICC(y_te,y_hat)
print 'avr. CORR:',cor.metrics.ICC(y_te,y_hat).mean()
print 
print cor.metrics.CORR(y_te,y_hat)
print 'avr. ICC:',cor.metrics.ICC(y_te,y_hat).mean()
