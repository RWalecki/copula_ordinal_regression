import theano.tensor as TT

def sigmoid(nu, sigma, z):
    return TT.nnet.sigmoid((z-nu)/sigma)

def normcdf(nu, sigma, z):
    return 0.5 * ( 1 + TT.erf( (z-nu) / (sigma * (2**0.5) ) ) )
