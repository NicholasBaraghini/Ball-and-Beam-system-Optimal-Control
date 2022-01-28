## Libraries
import numpy as np
import matplotlib
import system_dynamic
import optcon


# STAGE COST FUNCTION

def Stage_Cost(xx, uu, xx_ref, uu_ref, params):
    # INPUTS:
    #   - xx  : system state at current time t
    #   - uu  : input at current time t
    #   - xx_ref: reference state at time t
    #   - uu_ref: reference input at time t
    #   - params: list of parameters

    QQ = params['QQ']
    RR = params['RR']

    nx = np.shape(xx_ref)[0]  # state vector dimension
    nu = np.shape(uu_ref)[0]  # input vector dimension

    xx = np.reshape(xx, (nx,1))
    uu = np.reshape(uu, 1)
    uu_ref = np.reshape(uu_ref, 1)

    state_err = (xx - xx_ref)
    input_err = (uu - uu_ref)

    # input_err = np.reshape(input_err, 1)

    L_t = np.reshape(system_dynamic.dot3(state_err.T, QQ, state_err),1) + RR * (input_err ** 2)  # cost function evaluated at time t

    # GRADIENTS
    DLx = 2 * np.matmul(QQ, xx) - 2 * np.matmul(QQ, xx_ref)
    DLu = 2 * RR * uu - 2 * RR * uu_ref

    # 2nd order GRADIENTS
    DLxx = 2 * QQ
    DLuu = 2 * RR
    DLux = np.array(np.zeros((nu, nx)))

    # OUTPUTS: (the function returns an output dictionary with the follows entries)
    #   - cost_t : cost evaluated at time t
    #   - Dlx    : gradient of the cost w.r.t. the system state at time t
    #   - Dlu    : gradient of the cost dynamics w.r.t. the input at time t
    #   - DLxx   : hessian w.r.t. the system state at time t
    #   - DLux   : hessian w.r.t. ux
    #   - Dluu   : hessian w.r.t. the input at time t
    out = {
        'cost_t': L_t,
        'DLx': DLx,
        'DLu': DLu,
        'DLxx': DLxx,
        'DLux': DLux,
        'DLuu': DLuu
    }

    return out


# %%
# TERMINAL COST FUNCTION

def Terminal_Cost(xx_T, xx_T_ref, params):
    # INPUTS:
    #   - xx_T  : system state at current final time T
    #   - xx_T_ref: reference state at final time T
    #   - params: list of parameters

    QQ_T = params['QQ_T']

    nx = np.shape(xx_T_ref)[0]  # number of rows of xx_T_ref
    xx_T = np.reshape(xx_T, (nx, 1))

    state_err = (xx_T - xx_T_ref)

    L_T = np.reshape(system_dynamic.dot3(state_err.T, QQ_T, state_err), 1)  # cost function evaluated at final time T

    # GRADIENTS
    DLx = 2 * np.matmul(QQ_T, xx_T) - 2 * np.matmul(QQ_T, xx_T_ref)

    # 2nd order GRADIENTS
    DLxx = 2 * QQ_T

    # OUTPUTS: (the function returns an output dictionary with the follows entries)
    #   - cost_T : cost evaluated at final time T
    #   - Dlx    : gradient of the cost w.r.t. the system state at final time T
    #   - DLxx   : hessian w.r.t. the system state at final time T

    out = {
        'cost_T': L_T,
        'DLx': DLx,
        'DLxx': DLxx,
    }

    return out
