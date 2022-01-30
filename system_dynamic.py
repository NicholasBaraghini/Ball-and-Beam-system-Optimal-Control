# Libraries
import numpy as np
import matplotlib


# USEFUL FUNCTIONS

def dot3(a, B, c):
    # INPUTS:
    #   a : row vector 1xN
    #   B : matrix NxN
    #   c : column vector Nx1
    H = np.matmul(B, c)
    d = np.matmul(a, H)

    return d


# SYSTEM DYNAMICS (Ball and Beam)

# SYSTEM DYNAMIC FUNCTION

def BB_Dynamics(xx, uu, pp, params):
    # INPUTS:
    #   - XX    : system state at current time t
    #   - uu    : input at current time t
    #   - pp    : tensor product term
    #   - params: list of parameters

    # PARAMETERS EXTRACTION:
    dt = params['dt']  # Step size - Forward Euler method
    gg = params['gg']  # gravitational acceleration [m/s^2]
    mm = params['mm']  # ball mass [kg]
    rr = params['rr']  # ball radius [m]
    ii = params['ii']  # ball inertia [kg*m^2]
    II = params['II']  # beam inertia [kg*m^2]
    LL = params['LL']  # beam length [m]

    # USEFUL VARIABLES
    nx = 4  # number of states
    nu = 1  # number of inputs

    xx_next = np.zeros((4, 1))

    xx = np.reshape(xx, 4)
    uu = np.reshape(uu, 1)
    pp = np.reshape(pp, 4)

    # useful notations
    d1 = (mm + ii / (rr ** 2)) ** (-1)
    d2 = (II + mm * (xx[0] ** 2)) ** (-1)
    d22 = (II + mm * (xx[0] ** 2))

    # SYSTEM DYNAMICS

    xx_dot = [xx[1],
              (mm * xx[0] * (xx[3] ** 2) - mm * gg * np.sin(xx[2])) * d1,
              xx[3],
              -(2 * mm * xx[0] * xx[1] * xx[3] + mm * gg * xx[0] * np.cos(xx[2]) - uu) / d22]

    xx_next[0] = xx[0] + xx_dot[0] * dt
    xx_next[1] = xx[1] + xx_dot[1] * dt
    xx_next[2] = xx[2] + xx_dot[2] * dt
    xx_next[3] = xx[3] + xx_dot[3] * dt

    # GRADIENTS

    fx1_4_num = (-(2 * mm * xx[1] * xx[3] + mm * gg * np.cos(xx[2])) * d22 + (
            2 * mm * xx[0] * xx[1] * xx[3] + mm * gg * xx[0] * np.cos(xx[2]) - uu) * (2 * mm * xx[0]))
    fx1_4_den = d22 ** 2

    # partial derivative w.r.t. xx[1]:
    fx1 = np.array([[1,
                     dt * (mm * (xx[3] ** 2) * d1),
                     0,
                     dt * (fx1_4_num / fx1_4_den)[0]]])

    # partial derivative w.r.t. xx[2]:
    fx2 = np.array([[dt,
                     1,
                     0,
                     dt * (-2 * mm * xx[0] * xx[3] * d2)]])

    # partial derivative w.r.t. xx[3]:
    fx3 = np.array([[0,
                     dt * (-mm * gg * np.cos(xx[2]) * d1),
                     1,
                     dt * (mm * gg * xx[0] * np.sin(xx[2]) * d2)]])

    # partial derivative w.r.t. xx[4]:
    fx4 = np.array([[0,
                     dt * (2 * mm * xx[0] * xx[3] * d1),
                     dt,
                     1 - dt * (2 * mm * xx[0] * xx[1] * d2)]])

    # Jacobian of the system dynamics:
    fx = np.concatenate((fx1, fx2, fx3, fx4))

    # partial derivative w.r.t. the input:
    fu = np.array([[0,
                    0,
                    0,
                    dt * d2]]).T

    # SECOND ORDER GRADIENTS
    pfxx = np.zeros((nx, nx))
    pfux = np.zeros((nu, nx))
    pfuu = np.zeros((nu, nu))

    # useful notations
    fx1x1_4_num = (2 * mm * (2 * mm * xx[0] * xx[1] * xx[3] + mm * gg * xx[0] * np.cos(xx[2]) - uu) * (d22 ** 2)) - (
            fx1_4_num * (4 * d22 * mm * xx[0]))
    fx1x1_4_den = d22 ** 4

    # 1st row of the second derivative matrix nx*nx
    pfxx[0, 0] = pp[3] * (fx1x1_4_num / fx1x1_4_den) * dt
    pfxx[0, 1] = pp[3] * (-2 * mm * xx[0] * xx[3]) * (d2 ** 2) * dt
    pfxx[0, 2] = pp[3] * (mm * gg * xx[0] * np.sin(xx[2])) * (d2 ** 2) * dt
    pfxx[0, 3] = pp[1] * (2 * mm * xx[3] * d1 * dt) - pp[3] * (2 * mm * xx[0] * xx[1]) * (d2 ** 2) * dt

    # 2nd row of the second derivative matrix nx*nx
    pfxx[1, 0] = pp[3] * (2 * mm * xx[3] * (d22 - 2 * II)) * (d2 ** 2) * dt
    pfxx[1, 1] = 0
    pfxx[1, 2] = 0
    pfxx[1, 3] = pp[3] * (-2 * mm * xx[0] * d2) * dt

    # 3rd row of the second derivative matrix nx*nx
    pfxx[2, 0] = pp[3] * (-mm * gg * np.sin(xx[2]) * (d22 - 2 * II)) * (d2 ** 2) * dt
    pfxx[2, 1] = 0
    pfxx[2, 2] = pp[1] * (mm * gg * np.sin(xx[2]) * d1 * dt) + pp[3] * (mm * gg * xx[0] * np.cos(xx[2]) * d2 * dt)
    pfxx[2, 3] = 0

    # 4th row of the second derivative matrix nx*nx
    pfxx[3, 0] = pp[1] * (2 * mm * xx[3] * d1 * dt) + pp[3] * (2 * mm * xx[1] * (d22 - 2 * II)) * (d2 ** 2) * dt
    pfxx[3, 1] = pp[3] * (-2 * mm * xx[0] * d2) * dt
    pfxx[3, 2] = 0
    pfxx[3, 3] = pp[1] * (2 * mm * xx[0] * d1) * dt

    # pfuu has all null elements

    # pfux has only one non-null element
    pfux[0, 0] = pp[3] * (- 2 * mm * xx[0]) * (d2 ** 2) * dt

    # OUTPUTS: (the function returns an output dictionary with the follows entries)
    #   - xx_next : system state at state (t+1)
    #   - fx     : gradient of the system dynamics w.r.t. the system state at time t
    #   - fu     : gradient of the system dynamics w.r.t. the input at time t
    #   - pfxx   : tensor product within the dynamics function seconnd order derivative w.r.t. the state, given vector pp
    #   - pfux   : tensor product within the dynamics function seconnd order derivative w.r.t. input and state, given vector pp
    #   - pfuu   : tensor product within the dynamics function second order derivative w.r.t. input, given vector pp
    out = {
        'xx_next': xx_next,
        'fx': fx,
        'fu': fu,
        'pfxx': pfxx,
        'pfux': pfux,
        'pfuu': pfuu
    }

    return out
