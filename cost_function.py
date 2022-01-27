## Libraries
import numpy as np
import matplotlib
import system_dynamic
import optcon


# STAGE COST FUNCTION

def Stage_Cost(xx, uu, xx_ref, uu_ref, params):
  # INPUTs:
  #   - xx  : system state at current time t
  #   - uu  : input at current time t
  #   - xx_ref: referance state at time t
  #   - uu_ref: referance input at time t
  #   - params: list of parameters

  QQ = params['QQ'];
  RR = params['QQ_T'];

  nx = np.shape(xx_ref)[0] # state vector dymension
  nu = np.shape(uu_ref)[0] # input vector dymension

  state_err = (xx - xx_ref);
  input_err = (uu - uu_ref);

  L_t = system_dynamic.dot3(state_err.T, QQ, state_err) + system_dynamic.dot3(input_err.T, RR, input_err); # cost function evaluated at time t

  # GRADIENTs
  DLx = 2*np.matmul(QQ,xx) - 2*np.matmul(QQ,xx_ref);
  DLu = 2*np.matmul(RR,uu) - 2*np.matmul(RR,uu_ref);

  # 2nd order GRADIENTs
  DLxx = 2*QQ;
  DLuu = 2*RR;
  DLux = np.array(np.zeros((nu,nx)));

  # OUTPUTs: (the fucnction returns an output dictionary with the follows entries)
  #   - cost_t : cost evaluated at time t
  #   - Dlx    : gradient of the cost w.r.t. the system state at time t
  #   - Dlu    : gradient of the cost dynamics w.r.t. the input at time t
  #   - DLxx   : hessian w.r.t. the system state at time t
  #   - DLux   : hessian w.r.t. ux
  #   - Dluu   : hessian w.r.t. the input at time t
  out = {
         'cost_t':L_t,
         'DLx':Dlx,
         'DLu':DLu,
         'DLxx':Dlxx,
         'DLux':Dlux,
         'Dluu':DLuu
        };

  return out;

#%%
# TERMINAL COST FUNCTION

def Terminal_Cost(xx_T, xx_T_ref, params):
  # INPUTs:
  #   - xx_T  : system state at current final time T
  #   - xx_T_ref: referance state at final time T
  #   - params: list of parameters

  QQ_T = params['QQ_T']

  nx = np.shape(xx_T_ref)[0] # number of rows of xx_T_ref

  state_err = (xx_T - xx_T_ref);

  L_T = system_dynamic.dot3(state_err.T, QQ, state_err); # cost function evaluated at final time T

  # GRADIENTs
  DLx = 2*np.matmul(QQ_T,xx_T) - 2*np.matmul(QQ_T,xx_T_ref);

  # 2nd order GRADIENTs
  DLxx = 2*QQ_T;

  # OUTPUTs: (the fucnction returns an output dictionary with the follows entries)
  #   - cost_T : cost evaluated at final time T
  #   - Dlx    : gradient of the cost w.r.t. the system state at final time T
  #   - DLxx   : hessian w.r.t. the system state at final time T

  out = {
         'cost_T':L_T,
         'DLx':Dlx,
         'DLxx':Dlxx,
        };

  return out;