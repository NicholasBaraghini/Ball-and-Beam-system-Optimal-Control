# Libraries
import numpy as np
import matplotlib
import cost_function
import system_dynamic as sd


# DDP Algorithm Components evaluated at k-th iteration
def DDP_comp_t_k(kk, xx, uu, xx_ref, uu_ref, descent, TT, params):
    # INPUTS      : system state tensor
    #   - uu      : input tensor
    #   - xx_ref  : system state reference matrix
    #   - uu_ref  : input reference matrx
    #   - descent : descent at the k-th iteration to be updated
    #   - TT      : final step T
    #   - params  : parameters dictionary

    nx = np.shape(xx_ref)[0]  # state vector dimension
    nu = np.shape(uu_ref)[0]  # input vector dimension

    # Initializations
    KK = np.zeros((nu, nx, TT))
    SS = np.zeros((nu, TT))  # sigma
    pp = np.zeros((nx, TT))
    PP = np.zeros((nx, nx, TT))

    for tt in range(TT - 2, 0, -1):
        uu_tk = uu[:, tt:tt + 1, kk:kk + 1]
        uu_ref_tt = uu_ref[:, tt:tt + 1]

        xx_tk = xx[:, tt:tt + 1, kk:kk + 1]

        xx_ref_tt = xx_ref[:, tt:tt + 1]

        pp_next = pp[:, tt + 1:tt + 2]
        SS_tt = SS[:, tt:tt + 1]

        # System dynamics ar time t k-th iteration
        dyn = sd.BB_Dynamics(xx_tk, uu_tk, pp_next, params)
        # stage cost at time t k-th iteration
        stC = cost_function.Stage_Cost(xx_tk, uu_tk, xx_ref_tt, uu_ref_tt, params)
        # terminal cost at time t k-th iteration
        trC = cost_function.Terminal_Cost(xx_tk, xx_ref_tt, params)

        # Gain Computation

        PP_tt = np.reshape(PP[:, :, tt + 1:tt + 2], (4, 4))
        fu = np.reshape(dyn['fu'], (4, 1))
        fx = dyn['fx']
        KK_tt = np.reshape(KK[:, :, tt:tt + 1], (1, 4))
        KK_inv_tt = np.reshape(np.linalg.pinv(KK[:, :, tt:tt + 1]), (1, 4))

        KS_dir_term = stC['DLuu'] + sd.dot3(fu.T, PP_tt, fu) + dyn['pfuu']
        KS_inv_term = np.linalg.inv(KS_dir_term)  # inverse factor of the DDP gain formula
        KK_dir_term = stC['DLux'] + sd.dot3(fu.T, PP_tt, fx) + dyn['pfux']  # second factor of the DDP gain formula

        KK[:, :, tt:tt + 1] = -np.matmul(KS_inv_term, KS_dir_term)

        # Sigma Computation
        SS_dir_term = stC['DLu'] + np.matmul(fu.T, pp_next)  # second factor of the DDP sigma formula

        SS[:, tt:tt + 1] = -np.matmul(KS_inv_term, SS_dir_term)

        # PP update
        PP_1_term = stC['DLxx'] + sd.dot3(fx.T, PP_tt, fx) + dyn['pfxx']  # PP first term (DDP formula)
        PP_2_term = sd.dot3(KK_inv_tt.T, KS_dir_term, KK_tt)  # PP second term (DDP formula)

        PP[:, :, tt:tt + 1] = np.reshape(PP_1_term - PP_2_term, (4, 4, 1))
        PP[:, :, TT:TT + 1] = np.reshape(trC['DLxx'], (4, 4, 1))

        # pp update
        pp_1_term = stC['DLx'] + np.matmul(fx.T, pp_next)  # PP first term (DDP formula)
        pp_2_term = sd.dot3(KK_inv_tt.T, KS_dir_term, SS_tt)  # PP second term (DDP formula)

        pp[:, tt:tt + 1] = np.reshape(pp_1_term - pp_2_term, (4, 1))
        pp[:, TT:TT + 1] = trC['DLx']

        # Descent Direction Computation
        descent = descent - np.matmul(SS_tt.T, SS_tt)

    # OUTPUTS:
    #   - KK      :
    #   - Sigma   :
    #   - PP      :
    #   - pp      :
    #   - descent :

    out = {
        'KK': KK,
        'Sigma': SS,
        'PP': PP,
        'pp': pp,
        'descent': descent
    }

    return out


# %%
# ARMIJO's Function

def Armijo(kk, xx, uu, xx_init, xx_ref, uu_ref, TT, cost, descent, cc, beta, Sigma, KK, pp, params):
    # INPUTS:
    #   - kk       : actual iteration
    #   - xx_init  : system initial state at time t=0
    #   - xx_ref   : reference state vector
    #   - uu_ref   : reference input vector
    #   - TT       : final step T
    #   - cost     : cost vector
    #   - cc       :
    #   - beta     : step size of the Armijo's algorithm
    #   - Sigma    : Control Affine Element from the DDP algorithm
    #   - KK       : Feedback Gain Matrix frm the DDP algorithm
    #   - pp       : vector p from the DDP algorithm
    #   - params   : parameter dictionary

    nx = np.shape(xx_ref)[0]  # state vector dymension
    nu = np.shape(uu_ref)[0]  # input vector dymension

    gammas = np.array([[1]])  # initial step size
    armijo_cost = np.array([[]])  # cost for armijo function, update each iteration

    # temporary variable initialization
    xx_temp = np.zeros((nx, TT))
    uu_temp = np.zeros((nu, TT))

    # ARMIJO's LOOP
    while True:
        xx_temp[:, 0:1] = xx_init
        cost_temp = 0

        for tt in range(0, TT - 1):
            uu_tk = np.reshape(uu[:, tt:tt + 1, kk:kk + 1], (1, 1))
            uu_ref_tt = uu_ref[:, tt:tt + 1]
            uu_temp_tt = uu_temp[:, tt:tt + 1]

            xx_tk = np.reshape(xx[:, tt:tt + 1, kk:kk + 1], (4, 1))

            xx_ref_tt = xx_ref[:, tt:tt + 1]
            xx_temp_tt = xx_temp[:, tt:tt + 1]

            pp_next = pp[:, tt + 1:tt + 2]

            KK_tt = np.reshape(KK[:, :, tt:tt + 1], (1, 4))

            # temporary input control computation
            uu_temp[:, tt:tt + 1] = uu_tk + gammas[-1] * Sigma[:, tt:tt + 1] + np.matmul(KK_tt, (xx_temp_tt - xx_tk))
            # temporary system dynamics computation
            xx_temp[:, tt + 1:tt + 2] = sd.BB_Dynamics(xx_temp_tt, uu_temp_tt, pp_next, params)['xx_next']
            # stage cost computation
            cost_dummy = cost_function.Stage_Cost(xx_temp_tt, uu_temp_tt, xx_ref_tt, uu_ref_tt, params)['cost_t']

            # cost sum at for each stage cost in time [0,T-1]
            cost_temp += cost_dummy

        # cost sum at for each stage cost in time [0,T]
        cost_temp += cost_function.Terminal_Cost(xx_temp_tt, xx_ref_tt, params)['cost_T']

        # Cost structure collecting the cost registered for each gamma (step size)
        armijo_cost = np.append(armijo_cost, np.reshape(cost_temp, 1))

        if armijo_cost[-1] <= (cost[kk]) + cc * gammas[-1] * descent:
            return gammas

        # Structure collecting all the gamma computed not satisfying the Armijo's condition
        gammas = np.append(gammas, beta * gammas[-1])

    # OUTPUT:
    #   - gammas : array containing all the step sizes computed by the Armijo's
    #              Loop, gamma[-1] (the last element of the array) is the valid
    #              step sized should be considered


# Function implemented to update the Trajectory state and input
def Trajectory_Update(kk, xx, uu, xx_ref, uu_ref, xx_init, TT, cost, gamma, Sigma, KK, pp, params):
    # INPUTS:
    #   - kk       : actual iteration
    #   - xx       : system state tensor
    #   - uu       : input tensor
    #   - xx_ref   : reference state vector
    #   - uu_ref   : reference input vector
    #   - xx_init  : system initial state at time t=0
    #   - TT       : final step T
    #   - cost     : cost vector
    #   - gamma    : Step size
    #   - Sigma    : Control Affine Element from the DDP algorithm
    #   - KK       : Feedback Gain Matrix frm the DDP algorithm
    #   - pp       : vector p from the DDP algorithm
    #   - params   : parameter dictionary

    xx[:, 0:1, kk + 1:kk + 2] = np.reshape(xx_init, (4, 1, 1))

    for tt in range(0, TT - 1):
        uu_tk = np.reshape(uu[:, tt:tt + 1, kk:kk + 1],(1,1))
        uu_ref_tt = uu_ref[:, tt:tt + 1]
        uu_next_tt = uu[:, tt:tt + 1, kk + 1:kk + 2]

        xx_tk = np.reshape(xx[:, tt:tt + 1, kk:kk + 1],(4,1))
        xx_ref_tt = xx_ref[:, tt:tt + 1]
        xx_next_tt = np.reshape(xx[:, tt:tt + 1, kk + 1:kk + 2], (4,1))

        pp_next = pp[:, tt + 1:tt + 2]

        KK_tt = np.reshape(KK[:, :, tt:tt + 1], (1, 4))

        # Input vector update at time t
        uu_temp = uu_tk + gamma * Sigma[:, tt:tt + 1] + np.matmul(KK_tt, (xx_next_tt - xx_tk))
        uu[:, tt:tt + 1, kk + 1:kk + 2] = np.reshape(uu_temp, (1, 1, 1))

        # State vector update at time t
        xx_temp = sd.BB_Dynamics(xx_next_tt, uu_next_tt, pp_next, params)['xx_next']
        xx[:, tt + 1:tt + 2, kk + 1:kk + 2] = np.reshape(xx_temp, (4, 1, 1))

        # Cost Function Increment contribution at time t
        cost[kk + 1] = cost[kk + 1] + cost_function.Stage_Cost(xx_next_tt, uu_next_tt, xx_ref_tt, uu_ref_tt, params)[
            'cost_t']

    cost[kk + 1] = cost[kk + 1] + cost_function.Terminal_Cost(xx_next_tt, xx_ref_tt, params)['cost_T']

    out = {
        'xx': xx,
        'uu': uu,
        'cost': cost
    }

    return out
