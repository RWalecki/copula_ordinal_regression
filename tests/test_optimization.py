import unittest
import copula_ordinal_regression.tespo as tespo
import numpy as np
from examples import log_reg

X,y  = log_reg.rand_dset()
p0   = log_reg.init(X,y)
pred = tespo.compile(log_reg.pred, [p0, X], jac=False)
loss, grad = tespo.compile(log_reg.loss, [p0, X, y], jac=True)


class testcase:

    def test_default(self):

        p1, res = tespo.optimize(
            fun=loss,
            p0=p0,
            jac=grad,
            callback='default',
            args=(X, y),
            method='BFGS',
            options = {'maxiter': 20, 'disp': 0},
        )

        loss_0  = tespo.exe(loss, [p0, X, y])
        loss_1  = tespo.exe(loss, [p1, X, y])

        assert loss_0 > loss_1




if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__file__, env={'NOSE_NOCAPTURE' : 1})
