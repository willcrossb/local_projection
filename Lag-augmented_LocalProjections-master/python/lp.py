#function [ir, ir_varcov, betahat, betahat_varcov, res, X] = lp(Y,num_lags,horz,resp_ind,se_setting,no_const)

from statsmodels.tsa.tsatools import lagmat
import numpy as np
from linreg import * 

Y_in = np.array([1,2,1,3,4,2,1,2,3,2,4,1])
X_in = np.array([4,4,5,3,2,4,5,6,7,2,3,2])
Y_in = np.transpose(Y_in)
X_in = np.transpose(X_in)


def lp(Y, num_lags, horz, resp_ind_in, se_setting, no_const):
    # Covariate matrix

    print("inside function call")
    print(resp_ind_in)
    try:
        resp_ind = resp_ind_in.astype(int)
    except:
        resp_ind = resp_ind_in

    print(resp_ind)
    for i in range(num_lags+1):
        if i == 0:
            Y_lag = Y 
        else:
            nan_app = np.empty([i,Y.shape[1]])
            print("this is the nans")
            print(nan_app)
            append = np.vstack([nan_app, Y[0:Y.shape[0]-i,:]])
            print(append)
            #Y_lag = np.stack([Y_lag,append],axis =1)
            Y_lag = np.hstack([Y_lag,append])
            print("this is append")
            print(append)

        
    print(Y_lag)
    #lagged_matrix = np.column_stack([X[i:X.shape[0]-max(lags)+i] for i in lags])

    print("this is the lagmatrix")
    print(Y_lag)
    X = Y_lag[num_lags:Y_lag.shape[0]-horz,:]
    print()
    Y_reg = Y[num_lags+horz:,(resp_ind-1)]
    # Local projection
    print("dimensions of x then Y")
    print(X.shape)
    print("y")
    print(Y_reg.shape)
    print(Y_lag.shape)
    print("THIS IS THE MOST IMPORTANT Y REG")
    print(Y_reg)

    betahat, betahat_varcov, res, X = linreg(Y[num_lags+horz:,(resp_ind-1)], X, se_setting, no_const)
    
    n = Y.shape[1]
    m = 1
    #ISSUE HERE with the m line of code and the ir line of code
    ir = betahat[0:n]
    ir_varcov = betahat_varcov[:m*n,:m*n]
    
    return ir, ir_varcov, betahat, betahat_varcov, res, X


#calling function starts here

df = np.stack((Y_in,X_in),axis = 1)
print(df.shape)

(ir, ir_varcov, betahat, betahatvarcov, res, X)= lp(df, 2, 1, 1, False, False)

#print("results begin here")
#print(X)
