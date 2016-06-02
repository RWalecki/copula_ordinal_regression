import unittest
import copula_ordinal_regression.metrics as metrics
import numpy as np
import copy

methods = [[name,val] for name, val in metrics.__dict__.iteritems() if callable(val)]


class testcase:

    def test_default(self):

        y_lab = np.random.randint(0, 9, (4, 1000)).T
        y_lab[y_lab>0]=1
        y_lab = y_lab==0
        y_hat = copy.copy(y_lab)

        for n,f in methods:
            if n[0]=='_':continue
            score = f(y_hat,y_lab).mean()
            isinstance(score, float)

        for i in range(2000):
            idx_x = np.random.randint(0, y_hat.shape[0], 10)
            idx_y = np.random.randint(0, y_hat.shape[1], 10)
            y_hat[idx_x, idx_y] = 1


        for n,f in methods:
            if n[0]=='_':continue
            score = f(y_hat,y_lab).mean()
            isinstance(score, float)






if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__file__, env={'NOSE_NOCAPTURE' : 1})
