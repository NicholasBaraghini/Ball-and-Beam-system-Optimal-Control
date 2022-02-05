from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt
def Cubic_Ref(time,params,TT,dim,plot = 0):

    t1= np.arange(0, time[0], params['dt'])
    Tf1 = time[0]
    L = params['LL']
    nx = dim[0]
    nu = dim[1]
    a2 = 9 / 8 * L
    a3 = -3 / 4 * L
    cycle1 = a2 * (t1 ** 2) + a3 * (t1 ** 3)
    cycle1_vel = 2 * t1 * a2 + 3 * a3 * (t1 ** 2)

    Tf2 = time[1]
    t2 = np.arange(Tf1, Tf2, params['dt'])
    a0_2 = 3 / 8 * L
    a2_2 = -(21 / 8 * L) / ((Tf2 - Tf1) ** 2)
    a3_2 = (14 / 8 * L) / ((Tf2 - Tf1) ** 3)
    cycle2 = a0_2 + a2_2 * ((t2 - Tf1) ** 2) + a3_2 * ((t2 - Tf1) ** 3)
    cycle2_vel = 2 * (t2 - Tf1) * a2_2 + 3 * a3_2 * ((t2 - Tf1) ** 2)

    Tf3 = time[2]
    t3 = np.arange(Tf2, Tf3, params['dt'])
    a0_3 = -L / 2;
    a2_3 = (3 / 2 * L) / ((Tf3 - Tf2) ** 2);
    a3_3 = (- L) / ((Tf3 - Tf2) ** 3)
    cycle3 = a0_3 + a2_3 * ((t3 - Tf2) ** 2) + a3_3 * ((t3 - Tf2) ** 3)
    cycle3_vel = 2 * (t3 - Tf2) * a2_3 + 3 * a3_3 * ((t3 - Tf2) ** 2)

    tot_time = time[2]
    tt = np.arange(0, tot_time, params['dt'])
    traj = np.concatenate((cycle1.T, cycle2.T, cycle3.T))
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
    uu_ref[0, 0:TT] = params['mm']*params['gg'] * xx_ref[0,:]

    out = {
        'xx_ref': xx_ref,
        'uu_ref': uu_ref
    }
    return out

def Spline_Ref(x_point_ref, t_point_ref, TT, params, plot=0):
    #INPUTS:
    #  - x_point_ref : waypoints points on the x axes to be interpolated by the cubic spline
    #  - t_point_ref : times points corresponding to the waypoints
    #  - TT          : final step T
    #  - params      : parameters dictionary
    #  - plot        : flag used to display plot of the generated trajectory (default = 0)
    #OUTPUTS
    #  - xx_ref      : third-order polynomial reference trajectory's
    #  - uu_ref      : corresponding input torques, in quasi stationary condition, respect the generated curve

    spline_plot = plot          #flag dor displaying the plot
    nx = params['dim_X']        #dimension of states
    nu = params['dim_U']        #dimension of inputs

    tt = t_point_ref                  # times points
    xx = x_point_ref                  # waypoints

    cs = CubicSpline(tt, xx,bc_type=((1,0),(1,0)))      #function generating a cubic spline given waypoints, time points and velocity contraints
    xs = np.arange(0, params['tf'], params['dt'])       #times samples

    xx_ref_pos = cs(xs)               #reference position of the cubic spline
    xx_ref_vel = cs(xs, 1)            #reference velocity of the cubic spline
    xx_ref_acc = cs(xs, 2)            #reference acceleration of the cubic spline

    if spline_plot:
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(xs, xx_ref_pos, label="Spline", lw = 3)
        ax.plot(xs, xx_ref_vel, label="S'")
        ax.plot(xs, xx_ref_acc, label="S''")
        ax.legend(loc='lower left', ncol=1)
        plt.xlabel("t")
        plt.ylabel("p(t)")
        plt.title('Desired Evolution')
        plt.grid()
        plt.show
        fig.savefig('plot/spline_trajectory.jpg', transparent=True)
        plt.close(fig)





    #state reference definition
    xx_ref = np.zeros((nx, TT))
    xx_ref[0, 0:TT] = xx_ref_pos                                #reference position samples
    xx_ref[1, 0:TT] = xx_ref_vel                                #reference velocity samples

    # input reference definition
    uu_ref = np.zeros((nu, TT))
    uu_ref_qs = params['mm'] * params['gg'] * xx_ref_pos        # input torque in qs condition
    uu_ref[0, 0:TT] = uu_ref_qs                                 #reference inputs samples in qs condition

    out = {
        'xx_ref':xx_ref,
        'uu_ref':uu_ref
    }

    return out