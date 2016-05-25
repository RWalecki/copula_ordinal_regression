import unittest
import copula_ordinal_regression.tespo as tespo
import numpy as np
import log_reg

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


    def test_none(self):
        p1, res = tespo.optimize(
            fun=loss,
            p0=p0,
            jac=grad,
            args=(X, y),
            method='BFGS',
            options = {'maxiter': 20, 'disp': 0},
        )

    def test_user_counter(self):

        global glob_counter
        glob_counter = 0


        def _callback(pi):
            Y_hat = tespo.exe(pred, [pi, X])
            out = {}
            out['MSE']   = np.mean(Y_hat==y)
            out['Loss']  = tespo.exe(loss, [pi, X,y])
            opt = {'freq':10}
            return out, opt

        p1, res = tespo.optimize(
            fun=loss,
            p0=p0,
            jac=grad,
            callback=_callback,
            args=(X, y),
            method='BFGS',
            options = {'maxiter': 20, 'disp': 0},
        )


if __name__ == "__main__":
    import nose
    nose.run(defaultTest=__file__, env={'NOSE_NOCAPTURE' : 1})
