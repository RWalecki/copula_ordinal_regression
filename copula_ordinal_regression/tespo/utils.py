import numpy as np
from copy import deepcopy
import theano.tensor as TT

def make_theano_tensors(list_of_datasets):
    theano_tensor = [TT.dscalar(), TT.dvector(), TT.dmatrix(), TT.dtensor3()]
    res = []
    for d_set in list_of_datasets:
        dim = len(np.array(d_set).shape)
        theano_t = deepcopy(theano_tensor[dim])
        res.append(theano_t)
    return res

def para_2_vector(para):
    u = []

    keys=para.keys()
    keys.sort()
    for p in keys:
        if para[p].const:
            continue
        u.append(para[p].value.flatten(0))
    res = np.concatenate(u)
    return res

def vector_2_para(vec, para):
    res = deepcopy(para)
    u0 = 0

    keys=para.keys()
    keys.sort()
    for p in keys:
        if para[p].const:
            continue
        u1 = u0 + para[p].value.flatten().shape[0]
        tmp = vec[u0:u1].reshape(para[p].value.shape)
        res[p].value = tmp
        u0 = u1
    return res
