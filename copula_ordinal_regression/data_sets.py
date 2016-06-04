import gzip
import os
import numpy as np
pwd = os.path.join(os.path.dirname(__file__))

def load_disfa():
    dat = np.load(pwd+'/data/disfa.npz')
    X = np.vstack(dat['X'])
    y = np.vstack(dat['y'])
    S = np.hstack([[ii]*jj.shape[0] for ii,jj in zip(dat['S'],dat['y'])])
    return X,y,S

def load_fera2015():
    dat = np.load(pwd+'/data/fera2015.npz')
    X = np.vstack(dat['X'])
    y = np.vstack(dat['y'])
    S = np.hstack([[ii]*jj.shape[0] for ii,jj in zip(dat['S'],dat['y'])])
    return X,y,S

def load_shoulder_pain():
    dat = np.load(pwd+'/data/shoulder_pain.npz')
    X = np.vstack(dat['X'])
    y = np.vstack(dat['y'])
    S = np.hstack([[ii]*jj.shape[0] for ii,jj in zip(dat['S'],dat['y'])])
    return X,y,S

