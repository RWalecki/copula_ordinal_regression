import opengm
import numpy as np
import itertools

def ncll(X,F,E,state):
    cost = 0
    for vi in range(X.shape[0]):
        cost += X[vi,state[vi]]

    for e in range(len(E)):
        vi0,vi1= E[e]
        factor = F[e]
        cost += factor[state[vi0],state[vi1]]

    return cost

def APPROX(X,F,E,algorithm='lbp',MAP=False):
    res = []
    for i in range(X.shape[0]):
        res.append( approx(X[i], F[i], E, algorithm, MAP ) )
    return np.array(res)

def approx(X,F,E,algorithm='lbp',MAP=False):

    E = np.array(E,dtype=int)

    if algorithm == 'margins':
        pass
        #return: node/sum(node) and argmax(node)




    if algorithm == 'lbp':

        n_variales, n_states = X.shape

        numberOfStates = np.ones(n_variales, dtype=opengm.label_type)*n_states
        gm = opengm.gm(numberOfStates, operator='multiplier')

        for vi in range(X.shape[0]):
            unaryFuction = X[vi]
            gm.addFactor(gm.addFunction(unaryFuction), vi)

        for e in range(len(E)):
            vi0 = min(E[e])
            vi1 = max(E[e])
            factor = F[e]
            gm.addFactor(gm.addFunction(factor),[vi0,vi1])


        inf_algo=opengm.inference.BeliefPropagation(gm,parameter=opengm.InfParam(steps=10,damping=0.5,convergenceBound=0.001))

        inf_algo.infer()
        if MAP:
            return inf_algo.arg()
        else:
            res = []
            for i in range(n_variales):
                M = inf_algo.marginals(i)[0]
                M = M/M.sum()
                res.append(M)
        return np.array(res)


def get_range(X,F,E):
    n_F, n_S = X.shape
    c = []
    s = []
    for i in range(1000):
        state = np.random.randint(0,n_S,n_F)
        c.append( ncll(X,F,E,state) )
        s.append(state)
    idx = np.argmin(np.array(c))
    return s[idx]



def random_graph(AUs=5, Intensities=3, n_edges='all'):
    pdf = np.random.rand(AUs,Intensities)**2
    X = -np.log((pdf.T/pdf.T.sum(0)).T)

    tmp = []
    for i in itertools.combinations(range(AUs), 2):
        tmp.append(i)
    if n_edges=='all':
        E = np.array(tmp)
    else:
        np.random.shuffle(tmp)
        E = np.array(tmp[:n_edges])
    jpdf = np.random.rand(len(E),Intensities,Intensities)
    for i in range(len(E)):
        F = -np.log((jpdf.T/jpdf.T.sum(0)).T)

    return X,F,E

if __name__ == "__main__":
    X,F,E = random_graph(AUs=5,n_edges=6)
    print X.shape
    print F.shape
    print E.shape
    state = approx(X,F,E,'margins')
    print ncll(X,F,E,state)
    state = approx(X,F,E,'lbp')
    print ncll(X,F,E,state)
    state = approx(X,F,E,'bf')
    print ncll(X,F,E,state)
    state = approx(X,F,E,'ad3')
    print ncll(X,F,E,state)
    state = get_range(X,F,E)
