import theano.tensor as TT

def frank(u,v,d,cut=25):
    '''
    Frank Copula
    '''
    d = (TT.nnet.sigmoid(d)-0.5)*cut
    U = TT.exp(-d*u)-1
    V = TT.exp(-d*v)-1
    D = TT.exp(-d  )-1
    C = 1+U*V/D

    idx = TT.le(C,0).nonzero()
    C = TT.set_subtensor(C[idx],0)

    C = -1/(d) * TT.log(C)

    return C

def gumbel(u,v,d,cut=25):
    '''
    Gumbel Copula
    '''

    d = cut*(TT.nnet.sigmoid(d))-1
    U = (-TT.log(u))**d
    V = (-TT.log(v))**d
    P = TT.exp(-((U+V)**(1/d)))

    idx = TT.le(P,0).nonzero()
    P = TT.set_subtensor(P[idx],0)

    idx = TT.ge(P,1).nonzero()
    P = TT.set_subtensor(P[idx],1)

    return P

def indep( u,v,d=None):
    '''
    '''
    return u*v
