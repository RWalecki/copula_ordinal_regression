import theano as T
import theano.tensor as TT
import numpy as np
import scipy
import utils
from copy import deepcopy
import collections
glob_counter = 0


class parameter():
    '''
    '''
    def __init__(self, value, const=False):
        self.value = np.array(value, dtype=np.float32)
        self.const = const

def compile(fun, args, jac=False):
    def _fun(args):
        array = args[0]
        rest = args[1:]
        tmp_para = utils.vector_2_para(array, para)
        return fun(tmp_para, *rest)

    para = args[0]
    rest = args[1:]

    T_array = TT.dvector()
    T_rest = utils.make_theano_tensors(rest)
    args = [T_array]+T_rest

    fun_T = T.function(args, _fun(args),on_unused_input='warn')

    if not jac:
        return fun_T
    else:
        jac_fun = T.grad(_fun(args), args[0])
        jac_T = T.function(args, jac_fun,on_unused_input='warn')
        return fun_T, jac_T

def optimize(p0, fun, method='BFGS', args=None, jac=None, callback='default', options=None):
    glob_counter = 0

    def _callback_default(xi):
        global glob_counter
        if glob_counter % 10 == 0:
            count = str(glob_counter)
            while len(count)<4:count=' '+count
            print 'iter:', count, "%0.5e" % fun(xi, *args)
            pi = utils.vector_2_para(xi, p0)
        glob_counter += 1

    def _callback_none(xi):pass

        

    if callback == None:
        _callback = _callback_none

    if callback == 'default':
        _callback = _callback_default

    if callback is not 'default' and callback is not None:
        table, opt = callback(p0)
        table = collections.OrderedDict(sorted(table.items()))

        res = ''
        for item in table:
            while len(item)<11:item=item+' '
            res+=item[:10]+' '
        print 'Epoch:     ' + res
        print '-'*(len(table)*11+11)

        def _callback(xi):
            global glob_counter
            glob_counter += 1
            if glob_counter==1 or glob_counter%opt['freq']==0 or glob_counter==options['maxiter']:

                pi       = utils.vector_2_para(xi,p0)
                table, _ = callback(pi)
                table    = collections.OrderedDict(sorted(table.items()))
                line     = np.array([glob_counter]+table.values())

                for i in line:
                    s = str('%.4g' % i)
                    while len(s)<9:s=s+' '
                    print s+' ',
                print

    res = scipy.optimize.minimize(
        x0 = utils.para_2_vector(p0),
        fun = fun,
        method = method,
        callback = _callback,
        args = args,
        jac = jac,
        options = options
    )

    p1 = deepcopy( utils.vector_2_para(res['x'], p0) )
    return p1, res

def debug(fun, args):
    para = args[0]
    tmp = {}
    for p in para:
        tmp[p] = deepcopy(para[p])
        tmp[p].value = T.shared(para[p].value)


    rest = []
    for r in args[1:]:
        rest.append(T.shared(r))

    return fun(tmp, *rest)

def exe(fun, args):
    para = args[0]
    rest = args[1:]
    v = utils.para_2_vector(para)
    return fun(v, *rest)
