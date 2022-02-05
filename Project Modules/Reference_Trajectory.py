# local file path
lfp = 'C:/Users/Baraghini/PycharmProjects/Ball-and-Beam-system-Optimal-Control'
# Libraries
import numpy as np
import matplotlib
import sys

sys.path.insert(1, lfp + '/Project Modules')
from scipy.interpolate import CubicSpline
import numpy as np
import matplotlib.pyplot as plt


def Spline_Ref(x_point_ref, t_point_ref, TT, params, plot=0):
    # INPUTS:
    #  - x_point_ref : waypoints points on the x axes to be interpolated by the cubic spline
    #  - t_point_ref : times points corresponding to the waypoints
    #  - TT          : final step T
    #  - params      : parameters dictionary
    #  - plot        : flag used to display plot of the generated trajectory (default = 0)
    # OUTPUTS
    #  - xx_ref      : third-order polynomial reference trajectory's
    #  - uu_ref      : corresponding input torques, in quasi stationary condition, respect the generated curve

    spline_plot = plot  # flag dor displaying the plot
    nx = params['dim_X']  # dimension of states
    nu = params['dim_U']  # dimension of inputs

    tt = t_point_ref  # times points
    xx = x_point_ref  # waypoints

    cs = CubicSpline(tt, xx, bc_type=(
        (1, 0), (1, 0)))  # function generating a cubic spline given waypoints, time points and velocity contraints
    xs = np.arange(0, params['tf'], params['dt'])  # times samples

    xx_ref_pos = cs(xs)  # reference position of the cubic spline
    xx_ref_vel = cs(xs, 1)  # reference velocity of the cubic spline
    xx_ref_acc = cs(xs, 2)  # reference acceleration of the cubic spline

    if spline_plot:
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(xs, xx_ref_pos, label="Spline", lw=3)
        ax.plot(xs, xx_ref_vel, label="S'")
        ax.plot(xs, xx_ref_acc, label="S''")
        ax.legend(loc='lower left', ncol=1)
        plt.xlabel("t")
        plt.ylabel("p(t)")
        plt.title('Desired Evolution')
        plt.grid()
        plt.show
        fig.savefig(lfp + '/plot/spline_trajectory.jpg', transparent=True)
        plt.close(fig)

    # state reference definition
    xx_ref = np.zeros((nx, TT))
    xx_ref[0, 0:TT] = xx_ref_pos  # reference position samples
    xx_ref[1, 0:TT] = xx_ref_vel  # reference velocity samples

    # input reference definition
    uu_ref = np.zeros((nu, TT))
    uu_ref_qs = params['mm'] * params['gg'] * xx_ref_pos  # input torque in qs condition
    uu_ref[0, 0:TT] = uu_ref_qs  # reference inputs samples in qs condition

    out = {
        'xx_ref': xx_ref,
        'uu_ref': uu_ref
    }

    return out
