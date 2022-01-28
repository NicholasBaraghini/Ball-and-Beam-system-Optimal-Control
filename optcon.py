# Libraries
import numpy as np
import matplotlib
import cost_function
import PPdynamics as sd


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

    xx_Tk = xx[:, TT-1:TT, kk:kk + 1]  # shape (4,1,1)
    xx_ref_TT = xx_ref[:, TT-1:TT]  # shape (4,1)

    # terminal cost at time t k-th iteration
    trC = cost_function.Terminal_Cost(xx_Tk, xx_ref_TT, params)
    #cost_TT_kk = trC['cost_T']  # shape (4,4,1)
    Lx_TT_kk = trC['DLx']  # shape (4,1)
    Lxx_TT_kk = trC['DLxx']  # shape (4,4)

    PP[:, :, TT - 1:TT] = np.reshape(Lxx_TT_kk, (nx,nx, 1))
    pp[:, TT - 1:TT] = Lx_TT_kk

    for tt in range(TT - 2, -1, -1):

        # Parameters Definition
        uu_tk = uu[:, tt:tt + 1, kk:kk + 1] # shape (1,1,1)
        uu_ref_tt = uu_ref[:, tt:tt + 1] # shape (1,1)
        xx_tk = xx[:, tt:tt + 1, kk:kk + 1] # shape (4,1,1)
        xx_ref_tt = xx_ref[:, tt:tt + 1] # shape (4,1)
        pp_next = pp[:, tt + 1:tt + 2] # shape (4,1)
        SS_tt = SS[:, tt:tt + 1] # shape (1,1)

        # print('PP[:, :, tt + 1:tt + 2] :', PP[:, :, tt + 1:tt + 2])
        PP_next = np.reshape(PP[:, :, tt + 1:tt + 2], (nx, nx)) # shape (4,4)
        # print('PP[:, :, tt + 1:tt + 2] reshaped :', PP_next)
        # print('KK[:, :, tt:tt + 1] :', KK[:, :, tt:tt + 1])
        KK_tt = np.reshape(KK[:, :, tt:tt + 1], (1, nx)) # shape (1,4)
        # print('KK[:, :, tt:tt + 1] reshaped :', KK_tt)

        # System dynamics ar time t k-th iteration
        dyn = sd.BB_Dynamics(xx_tk, uu_tk, pp_next, params)
        # System Dynamics parameters extraction
        xx_next_kk = dyn['xx_next'] # shape (4,1)
        fx = dyn['fx'] # shape (4,4)
        # print('fu :', dyn['fu'])
        fu = np.reshape(dyn['fu'], (nx, 1)) # shape (4,1)
        # print('fu reshaped:', fu)
        pfxx_kk = dyn['pfxx']  # shape (4,4)
        pfux_kk = dyn['pfux']  # shape (1,4)
        pfuu_kk = dyn['pfuu']  # shape (1,1)

        # stage cost at time t k-th iteration
        stC = cost_function.Stage_Cost(xx_tk, uu_tk, xx_ref_tt, uu_ref_tt, params)
        # cost_tt_kk = stC['cost_t'] # shape (4,4,1)
        Lx_kk = stC['DLx'] # shape (4,1)
        Lu_kk = stC['DLu'] # shape (1,)
        Lxx_kk = stC['DLxx'] # shape (4,4)
        Lux_kk = stC['DLux'] # shape (1,4)
        Luu_kk = stC['DLuu'] # int


        # Gain Computation
        KS_dir_term = Luu_kk + sd.dot3(fu.T, PP_next, fu) + pfuu_kk  # term should be inverted
        KS_inv_term = np.linalg.inv(KS_dir_term)  # inverse factor of the DDP gain formula
        KK_dir_term = Lux_kk + sd.dot3(fu.T, PP_next, fx.T) + pfux_kk  # second factor of the DDP gain formula

        #print('KS_inv shape:', KS_inv_term.shape)
        #print('KK_dir shape:', KK_dir_term.shape)

        #print('KK: ',np.matmul(KS_inv_term, KK_dir_term))
        #print('KK reshappato: ', np.reshape(np.matmul(KS_inv_term, KK_dir_term), (1,4,1)))


        KK[:, :, tt:tt + 1] = - np.reshape(np.matmul(KS_inv_term, KK_dir_term), (1,nx,1))
        # Sigma Computation
        SS_dir_term = Lu_kk + np.matmul(fu.T, pp_next)  # second factor of the DDP sigma formula

        SS[:, tt:tt + 1] = -np.matmul(KS_inv_term, SS_dir_term)
        # PP update
        PP_1_term = Lxx_kk + sd.dot3(fx, PP_next, fx.T) + pfxx_kk  # PP first term (DDP formula)
        PP_2_term = sd.dot3(KK_tt.T, KS_dir_term, KK_tt)  # PP second term (DDP formula)
        #print('PP[:, :, tt:tt + 1] :',PP[:, :, tt:tt + 1] )
        PP[:, :, tt:tt + 1] = np.reshape(PP_1_term - PP_2_term, (nx, nx, 1))


        #print('PP[:, :, tt:tt + 1] RESHAPED :', PP[:, :, tt:tt + 1])

        # pp update
        pp_1_term = Lx_kk + np.matmul(fx, pp_next)  # PP first term (DDP formula)
        pp_2_term = sd.dot3(KK_tt.T, KS_dir_term, SS_tt)  # PP second term (DDP formula)

        #print('pp[:, tt:tt + 1] :', pp[:, tt:tt + 1])
        pp[:, tt:tt + 1] = np.reshape(pp_1_term - pp_2_term, (nx, 1))
        #print('pp[:, tt:tt + 1] RESHAPED :', pp[:, tt:tt + 1])
        # Descent Direction Computation
        descent -= - np.matmul(SS_tt.T, SS_tt)

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
    print('DDP_comp FINISHED')
    return out


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
    #print('gamma {},armcost {}, xx_temp{},uu_temp{}'.format(gammas.shape,armijo_cost.shape,xx_temp.shape, uu_temp.shape,))
    #wait = input('<ENTER>')
    # ARMIJO's LOOP
    iter = 0
    armijo_max_iter = 20
    while True:
        xx_temp[:, 0:1] = xx_init
        cost_temp = 0

        for tt in range(0, TT - 1):
            uu_tk = np.reshape(uu[:, tt:tt + 1, kk:kk + 1], (1,1))
            #print('uu',uu[:, tt:tt + 1, kk:kk + 1].shape, 'uu_tk:', uu_tk.shape)
            uu_ref_tt = uu_ref[:, tt:tt + 1]
            uu_temp_tt = uu_temp[:, tt:tt + 1]


            xx_tk = np.reshape(xx[:, tt:tt + 1, kk:kk + 1], (nx,1))
            #print('xx', xx[:, tt:tt + 1, kk:kk + 1], 'xx tk', xx_tk, 'shape:', type(xx_tk))
            xx_ref_tt = xx_ref[:, tt:tt + 1]
            xx_temp_tt = xx_temp[:, tt:tt + 1]

            pp_next = pp[:, tt + 1:tt + 2]

            KK_tt = np.reshape(KK[:, :, tt:tt + 1], (1,nx))
            #print('KK', KK, 'kk tt', KK_tt)

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

        #print(armijo_cost, '..',((cost[kk]) + cc * gammas[-1] * descent))
        if armijo_cost[-1] <= ((cost[kk]) + cc * gammas[-1] * descent):# or iter >= armijo_max_iter:
            print('ARMIJO found')
            return gammas

        iter += 1
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
    nx = np.shape(xx_ref)[0]  # state vector dymension
    nu = np.shape(uu_ref)[0]

    xx[:, 0:1, (kk + 1):(kk + 2)] = np.reshape(xx_init, (nx, 1, 1))

    for tt in range(0, TT - 2):
        uu_tk = np.reshape(uu[:, tt:(tt + 1), kk:(kk + 1)], (1, 1))
        uu_ref_tt = uu_ref[:, tt:(tt + 1)]
        uu_next_tt = uu[:, tt:(tt + 1), (kk + 1):(kk + 2)]

        xx_tk = np.reshape(xx[:, tt:(tt + 1), kk:(kk + 1)], (nx, 1))
        xx_ref_tt = xx_ref[:, tt:(tt + 1)]
        xx_next_tt = np.reshape(xx[:, tt:tt + 1, (kk + 1):(kk + 2)], (nx, 1))

        pp_next = pp[:, (tt + 1):(tt + 2)]

        KK_tt = np.reshape(KK[:, :, tt:(tt + 1)], (1, nx))

        # Input vector update at time t
        uu_temp = uu_tk + gamma * Sigma[:, tt:tt + 1] + np.matmul(KK_tt, (xx_next_tt - xx_tk))
        uu[:, tt:tt + 1, kk + 1:kk + 2] = np.reshape(uu_temp, (1, 1, 1))

        # State vector update at time t
        xx_temp = sd.BB_Dynamics(xx_next_tt, uu_next_tt, pp_next, params)['xx_next']
        xx[:, tt + 1:tt + 2, kk + 1:kk + 2] = np.reshape(xx_temp, (nx, 1, 1))

        # Cost Function Increment contribution at time t
        cost[kk+1] = cost[kk+1] + cost_function.Stage_Cost(xx_next_tt, uu_next_tt, xx_ref_tt, uu_ref_tt, params)[
            'cost_t']

    cost[kk+1] = cost[kk+1] + cost_function.Terminal_Cost(xx_next_tt, xx_ref_tt, params)['cost_T']

    out = {
        'xx': xx,
        'uu': uu,
        'cost': cost
    }

    return out
