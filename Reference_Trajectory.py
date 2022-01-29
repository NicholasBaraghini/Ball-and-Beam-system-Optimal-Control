from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt

def Cubic_Ref(time,params,plot,TT,dim):

    t1= np.arange(0, time[0], 0.0001)
    Tf1 = time[0]
    L = params['LL']
    a2 = 9 / 8 * L;
    a3 = -3 / 4 * L
    cycle1 = a2 * (t1 ** 2) + a3 * (t1 ** 3)
    cycle1_vel = 2 * t1 * a2 + 3 * a3 * (t1 ** 2)

    Tf2 = time[1]
    t2 = np.arange(Tf1, Tf2, 0.0001)
    a0_2 = 3 / 8 * L
    a2_2 = -(21 / 8 * L) / ((Tf2 - Tf1) ** 2);
    a3_2 = (14 / 8 * L) / ((Tf2 - Tf1) ** 3)
    cycle2 = a0_2 + a2_2 * ((t2 - Tf1) ** 2) + a3_2 * ((t2 - Tf1) ** 3)
    cycle2_vel = 2 * (t2 - Tf1) * a2_2 + 3 * a3_2 * ((t2 - Tf1) ** 2)

    Tf3 = time[2]
    t3 = np.arange(Tf2, Tf3, 0.0001)
    a0_3 = -L / 2;
    a2_3 = (3 / 2 * L) / ((Tf3 - Tf2) ** 2);
    a3_3 = (- L) / ((Tf3 - Tf2) ** 3)
    cycle3 = a0_3 + a2_3 * ((t3 - Tf2) ** 2) + a3_3 * ((t3 - Tf2) ** 3)
    cycle3_vel = 2 * (t3 - Tf2) * a2_3 + 3 * a3_3 * ((t3 - Tf2) ** 2)

    tot_time = time[0] + time[1] + time[2]
    tt = np.arange(0, tot_time, 0.0001)
    traj = np.concatenate((cycle1.T, cycle2[1:(len(cycle2))].T, cycle3[1:(len(cycle3))].T))
    traj_vel = np.concatenate((cycle1_vel.T, cycle2_vel[1:(len(cycle2_vel))].T, cycle3_vel[1:(len(cycle3_vel))].T))

    if plot:
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(tt, traj, 'o-', label='pos')
        ax.plot(tt, traj_vel, '.-', label='vel')
        ax.legend(loc='lower left', ncol=2)
        plt.grid()
        plt.show()

    xx_ref = np.zeros((dim[0], TT))
    xx_ref[0, 0:TT] = traj
    xx_ref[1, 0:TT] = traj_vel

    # input reference definition
    uu_ref = np.zeros((dim[1], TT))
    uu_ref[0, 0:TT] = uu_ref0

    out = {
        'xx_ref': xx_ref,
        'uu_ref': uu_ref
    }
    return out

def Spline_Ref(Time, params, x_point, t_point, plot, TT, dim):
    spline_plot = plot
    tt = t_point                             #tt = np.array([0, 1, 2.5, 3])
    L = params['LL']
    xx = x_point                             #xx = np.array([0, 3 / 8 * L, -L / 2, 0])
    cs = CubicSpline(tt, xx)
    xs = np.arange(0, Time, 0.0001)
    xx_ref0 = cs(xs)

    if spline_plot:
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(xs, cs(xs), label="Spline")
        ax.plot(xs, cs(xs, 1), label="S'")
        #ax.plot(xs, cs(xs, 2), label="S''")
        #ax.plot(xs, cs(xs, 3), label="S'''")
        ax.legend(loc='lower left', ncol=2)
        ax.plot(xs, xx_ref0)
        plt.grid()
        plt.show

    uu_ref0 = params['mm'] * params['gg'] * xx_ref0

    xx_ref = np.zeros((dim[0], TT));
    xx_ref[0, 0:TT] = xx_ref0;

    # input reference definition
    uu_ref = np.zeros((dim[1], TT));
    uu_ref[0, 0:TT] = uu_ref0;

    out = {
        'xx_ref':xx_ref,
        'uu_ref':uu_ref
    }

    return out