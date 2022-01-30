from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt


def Cubic_Ref(time, TT, dim, params, plot=0):
    mm = params['mm']
    gg = params['gg']

    t1 = np.arange(0, time[0], 0.0001)
    Tf1 = time[0]
    L = params['LL']
    a2 = 9 / 8 * L
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

    tot_time = time[2]
    tt = np.arange(0, tot_time, 0.0001)
    traj = np.concatenate((cycle1.T, cycle2[1:(len(cycle2))].T, cycle3[1:(len(cycle3))].T))
    traj_vel = np.concatenate((cycle1_vel.T, cycle2_vel.T, cycle3_vel.T))

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
    uu_ref0 = mm * gg * xx_ref[0, :]
    uu_ref[0, 0:TT] = uu_ref0

    out = {
        'xx_ref': xx_ref,
        'uu_ref': uu_ref
    }
    return out


def Spline_Ref(x_point_ref, t_point_ref, TT, params, plot=1):
    # Parameter Extraction
    dt = params['dt']
    Tf = params['Tf']
    nx = params['dim_x']
    nu = params['dim_x']
    mm = params['mm']
    gg = params['gg']

    spline_plot = plot
    tt = t_point_ref  # interpolation time point
    xx = x_point_ref  # interpolation position point

    cs = CubicSpline(tt, xx, bc_type=((1, 0), (1, 0)))  # SPLINE CREATION
    xs = np.arange(0, Tf, dt)  # sampling point of the spline

    xx_ref_pos = cs(xs)  # reference position from cubic spline
    xx_ref_vel = cs(xs, 1)  # reference velocity from cubic spline
    xx_ref_acc = cs(xs, 2)  # reference acceleration from cubic spline

    if spline_plot:
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(xs, xx_ref_pos, label="Spline")  # plot reference position
        ax.plot(xs, xx_ref_vel, label="S'")  # plot reference velocity
        ax.plot(xs, xx_ref_acc, label="S''")  # plot reference acceleration
        ax.legend(loc='lower left', ncol=1)
        plt.grid()
        plt.show

    xx_ref = np.zeros((nx, TT))
    xx_ref[0, 0:TT] = xx_ref_pos  # reference position from cubic spline
    xx_ref[1, 0:TT] = xx_ref_acc  # reference velocity from cubic spline

    # input reference definition
    uu_ref0 = mm * gg * xx_ref_pos
    uu_ref = np.zeros((nu, TT))
    uu_ref[0, 0:TT] = uu_ref0

    out = {
        'xx_ref': xx_ref,
        'uu_ref': uu_ref
    }

    return out
