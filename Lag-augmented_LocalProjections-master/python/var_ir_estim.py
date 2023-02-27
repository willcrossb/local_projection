from var_biascorr import * 
from lp import *
import numpy as np



def var_ir_estim(Y, p, p_estim, horzs, bias_corr, homosk, no_const):
    # VAR(p) least-squares estimates and delta method s.e.
    # allowing for lag augmentation

    # Inputs:
    # Y         T x n   data vector
    # p         1 x 1   lag length used for impulse response computations
    # p_estim   1 x 1   lag length used for estimation (p_estim >= p)
    # horzs     H x 1   horizons of interest
    # bias_corr bool    true: apply analytical bias correction (Pope, 1990)
    # homosk    bool    true: homoskedastic s.e., false: EHW s.e.
    # no_const  bool    true: omit intercept

    # Outputs:
    # irs           n x n x H           estimated impulse responses Theta_h at select horizons
    # irs_varcov    n^2 x n^2 x H       var-cov matrices of vec(Theta_h) at select horizons
    # Ahat_estim    n x np              VAR coefficient estimates [A_1,...,A_p] (possibly bias-corrected, possibly including intercept as last column)
    # res_estim     (T-p_estim) x n     estimation residuals

    T, n = Y.shape

    # One-step forecasting regression of Y_{t+1} on (Y_t, ..., Y_{t-p_estim+1})
    dn1,dn2,Ahat_estim, Ahat_estim_varcov, res_estim, dn3 = lp(Y, p_estim-1, 1, np.ones(n), homosk, no_const)

    #return ir, ir_varcov, betahat, betahat_varcov, res, X
    # If bias correction is desired...
    if bias_corr:
        print(res_estim.shape[0] - n*p_estim - 1 + no_const)
        Sigmahat = (res_estim.T @ res_estim) / (res_estim.shape[0] - n*p_estim - 1 + no_const) # Residual variance estimate
        Ahat_estim[:,:n*p] = var_biascorr(Ahat_estim[:,:n*p], Sigmahat, T)

    # Only use first p VAR coefficient matrices to compute impulse responses
    Ahat = Ahat_estim[:,:n*p]
    Ahat_varcov = Ahat_estim_varcov[:n**2*p,:n**2*p]

    
    # irs, jacob = var_ir(Ahat,horzs) # Compute impulse responses and Jacobian
    # nh = len(horzs)
    # irs_varcov = np.zeros((n**2,n**2,nh))
    # for h in range(nh):
    #     irs_varcov[:,:,h] = jacob[:,:,h] @ Ahat_varcov @ jacob[:,:,h].T # Var-cov for impulse response

    # return irs, irs_varcov, Ahat_estim, res_estim

    return Ahat_estim, res_estim

Y_in = np.array([1,2,1,3,4,2,1,2,3,2,4,1])
X_in = np.array([4,4,5,3,2,4,5,6,7,2,3,2])
Y_in = np.transpose(Y_in)
X_in = np.transpose(X_in)
df = np.stack((Y_in,X_in),axis = 1)
print(df.shape)
print("ones array")
print(np.ones(2,dtype=int))

(test1, test2) = var_ir_estim(df,1,1,np.ones(3,dtype=int),False,True,True)

