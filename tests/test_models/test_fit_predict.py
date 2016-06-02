import cPickle
import gzip
import numpy as np
import copula_ordinal_regression as cor

X, y, _ = cor.load_disfa()

class testcase:

    def test_mlr(self):
        clf = cor.COR(max_iter=50)
        clf.fit(X,y,debug=True)



if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__file__, env={'NOSE_NOCAPTURE' : 1})
