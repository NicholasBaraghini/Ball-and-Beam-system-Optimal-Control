from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

def Cubic_Ref(TT, params):
    t = np.arange(0, 1.05, 0.05)
    Tf1 = TT
    L = params['LL']
    a2 = 9 / 8 * L;
    a3 = -3 / 4 * L
    cycle1 = a2 * (t ** 2) + a3 * (t ** 3)
    cycle1_vel = 2 * t * a2 + 3 * a3 * (t ** 2)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t, cycle1, 'o-')
    ax.plot(t, cycle1_vel, '.-')
    plt.grid()
    plt.show()

    return