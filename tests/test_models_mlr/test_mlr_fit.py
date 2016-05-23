import cPickle
import gzip
import numpy as np
import copula_ordinal_regression as cor

class testcase:

    def test_default(self):

        dat = cPickle.load(gzip.open('./tests/data/disfa_slim.pklz','rb'))
        X = np.vstack(dat['X'])
        y = np.vstack(dat['y'])

        clf = cor.models.mlr(X, y, verbose=0, C=0, max_iter=10)

        y_hat = clf.fit(X,y).predict(X)

        mse_0 = np.mean((y-np.zeros_like(y))**2,0).mean()
        mse_p = np.mean((y-y_hat)**2,0).mean()
        mse_r = np.mean((y-np.random.randint(0,5,y.shape))**2,0).mean()

        assert mse_0 > mse_p
        assert mse_r > mse_p




if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__file__, env={'NOSE_NOCAPTURE' : 1})
