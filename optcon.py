# Libraries
import numpy as np
import matplotlib
import cost_function
import system_dynamic


# DDP Algorithm Components evaluated at k-th iteration
def DDP_comp_t_k(kk, xx, uu, xx_ref, uu_ref, params):
    # INPUTs:
    #   - kk     : iteration of evaluation
    #   - xx     : system state tensor
    #   - uu     : input tensor
    #   - xx_ref : system state reference matrix
    #   - uu_ref : input reference matrx
    #   - tt     : time of evaluation
    #   - TT     : final step T
    #   - params : parameters dictionary

    nx = np.shape(xx_ref)[0]  # state vector dymension
    nu = np.shape(uu_ref)[0]  # input vector dymension

    # Initializations
    KK = np.zeros((nu, nx, TT));
    SS = np.zeros((nu, TT));  # sigma
    pp = np.zeros((nn, TT));
    PP = np.zeros((nn, nn, TT));

    for tt in range(TT - 1, 0, -1):
        # System dymanics ar time t k-th iteration
        dyn = system_dynamic.BB_Dynamics(xx[:, tt, kk], uu[:, tt, kk], pp[:, tt + 1], params);
        # stage cost at time t k-th iteration
        stC = cost_function.Stage_Cost(xx[:, tt, kk], uu[:, tt, kk], xx_ref[:, tt], uu_ref[:, tt], params);
        # terminal cost at time t k-th iteration
        trC = cost_function.Terminal_Cost(xx[:, tt, kk], xx_ref[:, tt], params);

        # Gain Computation
        KS_dir_term = stC['luu'] + system_dynamic.dot3(dyn['fu'].T, PP[:, :, tt + 1], dyn['fu']) + dyn['pfuu'];
        KS_inv_term = np.linalg.inv(KS_dir_term);  # inverse factor of the DDP gain formula
        KK_dir_term = stC['lux'] + system_dynamic.dot3(dyn['fu'].T, PP[:, :, tt + 1], dyn['fx']) + dyn[
            'pfux'];  # second factor of the DDP gain formula

        KK = -np.matmul(KS_inv_term, KS_dir_term);

        # Sigma Computation
        SS_dir_term = stC['lu'] + np.matmul(dyn['fu'], pp[:, tt + 1]);  # second factor of the DDP sigma formula

        SS = -np.matmul(KS_inv_term, SS_dir_term);

        # PP update
        PP_1_term = stC['lxx'] + system_dynamic.dot3(dyn['fx'].T, PP[:, :, tt + 1], dyn['fx']) + dyn[
            'pfxx'];  # PP first term (DDP formula)
        PP_2_term = system_dynamic.dot3(np.linalg.inv(KK[:, :, tt]).T, KS_dir_term, KK[:, :, tt])  # PP second term (DDP formula)

        PP[:, :, tt] = PP_1_term - PP_2_term;

        PP[:, :, TT] = trC['DLxx'];

        # pp update
        pp_1_term = stC['lx'] + np.matmul(dyn['fx'].T, pp[:, tt + 1]);  # PP first term (DDP formula)
        pp_2_term = system_dynamic.dot3(np.linalg.inv(KK[:, :, tt]).T, KS_dir_term, SS[:, tt])  # PP second term (DDP formula)

        pp[:, tt] = pp_1_term - pp_2_term;

        pp[:, TT] = trC['DLx'];

        # Descent Direction Computation
        descent[kk] = descent[kk] - np.matmul(SS[:, tt].T, SS[:, tt]);

    # OUTPUTs:
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
    };

    return out;

# %%
# ARMIJO's Function

def Armijo(kk, xx_init, xx_ref, uu_ref, TT, cost, cc, beta, Sigma, KK, pp, params):
    # INPUTs:
    #   - kk       : actual iteration
    #   - xx_init  : system initial state at at time t=0
    #   - xx_ref   : referance state vector
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
        xx_temp[:, 0] = xx_init
        cost_temp = 0

        for tt in range(0, TT):
            # temporary input control computation
            uu_temp[:, tt] = uu[:, tt, kk] + gammas[-1] * sigma[:, tt] + np.matmul(KK[:, :, tt], (
                        xx_temp[:, tt] - xx[:, tt, kk]))  # !! KK e sigma missing
            # temporary system dynamics computation
            xx_temp[:, tt + 1] = system_dynamic.BB_Dynamics(xx_temp[:, tt], uu_temp[:, tt], pp[:, tt + 1], params)['xx_next']
            # stage cost computation
            cost_dummy = cost_function.Stage_Cost(xx_temp[:, tt], uu_temp[:, tt], xx_ref[:, tt], uu_ref[:, tt], params)['cost_t']

            # cost sum at for each stage cost in time [0,T-1]
            cost_temp += cost_dummy

        # cost sum at for each stage cost in time [0,T]
        cost_temp += cost_function.Terminal_Cost(xx_temp[:, TT], xx_ref[:, TT], params)['cost_T']

        # Cost structure collecting the cost registered for each gamma (step size)
        armijo_cost = np.concatenate(armijo_cost, cost_temp, axis=1)

        descent = DDP_comp_t_k(kk, xx, uu, xx_ref, uu_ref, params)['descent'];  # descent at time t, k-th iteration
        if (armijo_cost[-1] <= (cost[kk]) + cc * gammas[-1] * descent):
            return gammas

        # Stucture collecting all the gamma computed not satisfying the Armijo's condition
        gammas = np.concatenate(gammas, beta * gammas[-1], axis=1)

    # OUTPUT:
    #   - gammas : array containing all the step sizes computed by the Armijo's
    #              Loop, gamma[-1] (the last element of the array) is the valid
    #              step sized should be considered


# Function implemented to update the Trajectory state and input
def Trajectory_Update(kk, xx, uu, xx_ref, uu_ref, xx_init, TT, cost_kk, gamma, Sigma, KK, pp,  params):
    # INPUTs:
    #   - kk       : actual iteration
    #   - xx       : system state tensor
    #   - uu       : input tensor
    #   - xx_ref   : referance state vector
    #   - uu_ref   : reference input vector
    #   - xx_init  : system initial state at at time t=0
    #   - TT       : final step T
    #   - cost     : cost vector
    #   - gamma    : Step size
    #   - Sigma    : Control Affine Element from the DDP algorithm
    #   - KK       : Feedback Gain Matrix frm the DDP algorithm
    #   - pp       : vector p from the DDP algorithm
    #   - params   : parameter dictionary

    xx[:, 0, kk + 1] = xx_init;

    for tt in range(0, TT):
        # Input vector update at time t
        uu[:, tt, kk + 1] = uu[:, tt, kk] + gamma * Sigma[:, tt] + np.matmul(KK[:, :, tt],
                                                                             (xx[:, tt, kk + 1] - xx[:, tt, kk]));

        # State vector update at time t
        xx[:, tt + 1, kk + 1] = system_dynamic.BB_Dynamics(xx[:, tt, kk + 1], uu[:, tt, kk + 1], pp[:, tt + 1], params);

        # Cost Function Increment contribution at time t
        cost[kk + 1] = cost[kk + 1] + cost_function.Stage_Cost(xx[:, tt, kk + 1], uu[:, tt, kk + 1], xx_ref[:, tt], uu_ref[:, tt], params)['cost_t'];

    cost[kk + 1] = cost[kk + 1] + cost_function.Terminal_Cost(xx[:, TT, kk + 1], xx_ref[:, TT], params)['cost_T'];

    out = {
            'xx': xx,
            'uu': uu,
            'cost': cost
    }

    return out

