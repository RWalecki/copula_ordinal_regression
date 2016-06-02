import numpy as np
import theano as T
import theano.tensor as TT

def weights(y,type):
    if type==None:
        return np.ones_like(y)/np.float64(y.shape[0])
    if type=='balanced':
        lb = preprocessing.LabelBinarizer()
        lb.fit(y.flatten())
        res = np.zeros_like(y)
        for i in range(y.shape[1]):
            w_ = np.sum(lb.transform(y[:,i]),0)
            res[:,i] = w_[y[:,i]]
        return (np.float64(res)**-1.)

def expectation(P):
    states = TT.arange(P.shape[2]).dimshuffle('x','x',0)
    states = TT.extra_ops.repeat(states,P.shape[0],0)
    states = TT.extra_ops.repeat(states,P.shape[1],1)
    return TT.sum(P*states,axis=2)

def log_prob(P, realmin = 1e-20):
    '''
    numerical stabil log function
    for some reason theano needs a high realmin value
    '''

    # if prob is less than realmin, set it to realmin
    idx = TT.le(P,realmin).nonzero()
    P = TT.set_subtensor(P[idx],realmin)

    # if prob larger than 1-realmin, set it to 1-realmin
    idx = TT.ge(P,1-realmin).nonzero()
    P = TT.set_subtensor(P[idx],1-realmin)

    idx = TT.isnan(P).nonzero()
    P = TT.set_subtensor(P[idx],realmin)

    idx = TT.isinf(P).nonzero()
    P = TT.set_subtensor(P[idx],realmin)

    return TT.log(P)

def node_potn(pdf, y=None):
    '''
    pdf:     [AU X Frame X Label]

    node that there are M+1 Thresholds and they have to go from 0 to 1
    '''
    if y:
        y_ = y.T.astype('int8').flatten(1)
        pdf = pdf.T.flatten(2)
        idx = TT.arange(y_.shape[0])
        P = pdf[y_,idx]
        P = P.reshape(y.shape)
    else:
        P = pdf

    return -log_prob(P)

def edge_potn(pdf, copula, theta, edges,Y=None, shared_copula=False):
    '''
    '''
    cdf = TT.extra_ops.cumsum(pdf,axis=2)
    cdf = TT.concatenate((TT.zeros_like(cdf[:,:,[0]]),cdf),axis=2)

    def comp_jpdf(cdf, d, y=None):
        '''
        cdf : list of cdfs              [cdf_1, cdf_2]
        y : list of vecotr of labels    [y_1, y_1]
        '''
        idx = TT.arange(cdf.shape[1])
        if y:
            u_0 = cdf[0,idx,y[0]]
            u_1 = cdf[0,idx,y[0]+1]
            v_0 = cdf[1,idx,y[1]]
            v_1 = cdf[1,idx,y[1]+1]

            if shared_copula:
                pass
            else:
                d = d[y[0],y[1]]

            P =  copula(u_0,v_0,d)
            P -= copula(u_0,v_1,d)
            P -= copula(u_1,v_0,d)
            P += copula(u_1,v_1,d)
        else:

            cdf_0 = TT.extra_ops.repeat(cdf[0].dimshuffle(0,1,'x'),cdf[1].shape[1],2)
            cdf_1 = TT.extra_ops.repeat(cdf[1].dimshuffle(0,'x',1),cdf[0].shape[1],1)

            if shared_copula:
                j_cdf = copula(cdf_0,cdf_1,d)
                P = j_cdf[:,1:,1:] + j_cdf[:,:-1,:-1] - j_cdf[:,:-1,1:] - j_cdf[:,1:,:-1]
            else:
                u11 = cdf_0[:,1:,1:]
                u01 = cdf_0[:,:-1,1:]
                u10 = cdf_0[:,1:,:-1]
                u00 = cdf_0[:,:-1,:-1]

                v11 = cdf_1[:,1:,1:]
                v01 = cdf_1[:,:-1,1:]
                v10 = cdf_1[:,1:,:-1]
                v00 = cdf_1[:,:-1,:-1]

                d = d.dimshuffle('x',0,1)
                d = TT.extra_ops.repeat(d,cdf.shape[1],0)

                uv_11 = copula(u11,v11,d)
                uv_00 = copula(u00,v00,d)
                uv_01 = copula(u01,v01,d)
                uv_10 = copula(u10,v10,d)

                P = uv_00+uv_11-uv_10-uv_01
        return P


    cdf = cdf.dimshuffle(1,0,2)
    if Y != None:
        Y = Y.T.astype('int8')
    edges = edges.T.astype('int8')

    def inner_function(e, t, cdf):
        if Y == None:
            jpdf = comp_jpdf(cdf[e], t)
        else:
            jpdf = comp_jpdf(cdf[e], t, Y[e])
        return jpdf

    # inner_function(edges[0],theta[0],cdf)

    jpdf , _ = T.scan(
        fn=inner_function,
        sequences=[edges, theta],
        non_sequences=[cdf]
    )

    return -log_prob(jpdf)
    # return T.shared(0)
