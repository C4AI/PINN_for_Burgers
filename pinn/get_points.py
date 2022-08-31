import numpy as np
from scipy.io import loadmat

def domain(n,pars):

    t = np.random.uniform(low=0,high=pars['tf'],size=(n,1))
    x = np.random.uniform(low=pars['xi'],high=pars['xf'],size=(n,1))
    y = np.random.uniform(low=pars['yi'],high=pars['yf'],size=(n,1))

    X = np.hstack((t,x,y))

    return X

def solution(n,pars):

    data = loadmat(pars['solution_file'])

    [x, t, y] = np.meshgrid(data['x'],data['t'],data['y'])

    t = t.flatten().reshape(-1, 1)
    x = x.flatten().reshape(-1, 1)
    y = y.flatten().reshape(-1, 1)
    u = data['uref'].transpose((2,1,0)).flatten().reshape(-1, 1)
    v = data['vref'].transpose((2,1,0)).flatten().reshape(-1, 1)

    ind = np.random.choice(t.shape[0], size=n)

    X = np.hstack((t[ind],x[ind],y[ind]))
    Y = np.hstack((u[ind],v[ind]))

    return X, Y

def initialize_fourier_feature_network(n_transforms,sigma):
    B = {}
    #B['t'] = np.random.normal(scale=2*np.pi*sigma[0],size=n_transforms[0])
    B['t'] = np.array([2*np.pi*i for i in range(1,n_transforms[0]+1)])
    B['x'] = np.random.normal(scale=2*np.pi*sigma[1],size=n_transforms[1])
    B['y'] = np.random.normal(scale=2*np.pi*sigma[2],size=n_transforms[2])

    return B