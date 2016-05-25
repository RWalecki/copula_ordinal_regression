import cPickle
import gzip
import numpy as np
import copula_ordinal_regression as cor

dat = cPickle.load(gzip.open('./tests/data/disfa_slim.pklz','rb'))
X = np.vstack(dat['X'])
y = np.vstack(dat['y'])

def clr_fit_predict(clf):
    y_hat = clf.fit(X,y).predict(X)
    mse_0 = np.mean((y-np.zeros_like(y))**2,0).mean()
    mse_p = np.mean((y-y_hat)**2,0).mean()
    mse_r = np.mean((y-np.random.randint(0,5,y.shape))**2,0).mean()
    return (mse_0 > mse_p) and (mse_r > mse_p)



class testcase:

    def test_mlr(self):
        clf = cor.models.MLR(max_iter=50)
        assert clr_fit_predict(clf)

    def test_sor(self):
        clf = cor.models.SOR(max_iter=50)
        assert clr_fit_predict(clf)


if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__file__, env={'NOSE_NOCAPTURE' : 1})
