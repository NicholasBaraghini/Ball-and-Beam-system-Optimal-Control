import numpy as np

def dot3(a, B, c):
    # INPUTS:
    #   a : row vector 1xN
    #   B : matrix NxN
    #   c : column vector Nx1
    H = np.matmul(B, c)
    d = np.matmul(a,H)
    #GIMMY

    return d  # Returns the matrix product a*B*c

def BB_Dynamics(xx,uu,pp,params):
    ns = 2
    nu = 1

    xx_plus = np.zeros((ns,1))
    gg = 9.81
    ll = 1
    kk = 0
    mm = 1
    dt = 0.001

    xx = np.reshape(xx, 2)
    uu = np.reshape(uu, 1)
    pp = np.reshape(pp, 2)


    xx_plus[0] = xx[0] + dt* xx[1]
    xx_plus[1] = xx[1] + dt* ((gg/ll)*np.sin(xx[0]) - (kk/(mm * ll)) * xx[1] + uu / (mm * ll**2))

    #gradient
    fx = np.array([[1, dt * (gg/ll) * np.cos(xx[0])],
                   [dt, 1 + dt * (-kk / (mm*ll))]])

    fu = np.array([[0, dt * 1 / (mm * (ll**2))]])

    pfxx = np.zeros((ns,ns))
    pfuu = np.zeros((nu,nu))
    pfux = np.zeros((nu,ns))

    pfxx[0,0] = pp[1] * (-dt * gg/ll * np.sin(xx[0]))

    out = {
        'xx_next': xx_plus,
        'fx' : fx,
        'fu' : fu,
        'pfxx' : pfxx,
        'pfux' : pfux,
        'pfuu' : pfuu
    }
    return out
