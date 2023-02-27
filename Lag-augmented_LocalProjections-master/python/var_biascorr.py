import numpy as np
from scipy import linalg

def var_biascorr(A, Sigma, T):
    # Analytical bias correction for VAR(p) estimator
    # Pope (JTSA 1990), equation 9

    # Inputs:
    # A             n x np  original VAR(p) coefficient estimates [A_1,...,A_p]
    # Sigma         n x n   estimate of VAR(p) innovation variance
    # T             1 x 1   sample size

    # Outputs:
    # A_corr        n x np  bias-corrected coefficient estimates

    # Set up companion form: X_t = A*X_{t-1} + Z_t, where dim(X_t)=n*p
    (n, n_p) = A.shape
    test = np.hstack((np.identity(n_p-n), np.zeros((n_p-n,n))))

    #print(test)
    A_comp = np.vstack((A, np.hstack((np.identity(n_p-n), np.zeros((n_p-n,n))))))

    print(A_comp)

    #ISSUE
    # if np.max(np.abs(np.linalg.eig(A_comp)[0])) > 1: # If original point estimate is outside stationary region, do not bias correct
    #     A_corr = A
    #     return A_corr

    
    G = linalg.block_diag(Sigma, np.zeros((n,n_p-n))) # Var(Z_t)
    Gamma0 = linalg.solve_discrete_lyapunov(A_comp, G) # Var(X_t), requires Control System Toolbox
    # The following slower command does not require Control System Toolbox:
    # Gamma0 = np.reshape(np.linalg.solve(np.eye(np**2)-np.kron(A_comp,A_comp),G.flatten()),(np,np))
    # Bias correction formula
   
    #this line is a hassle due to lack of right divide function in python
    a_temp = A_comp.T
    b_temp = np.identity(n_p)-(A_comp.T @ A_comp.T)
    aux = np.linalg.inv(np.identity(n_p)-A_comp.T) + np.dot(a_temp, linalg.pinv(b_temp))

    print("first aux here")
    print(aux)
    lambdas = np.linalg.eig(A_comp)[0]
    for lamb in lambdas:
        aux = aux + lamb*np.linalg.inv(np.identity(n_p)-lamb * A_comp.T)

    
    b = G @ np.dot(aux, linalg.pinv(Gamma0))
    A_corr = A_comp + b/T # Bias-corrected companion form coefficients
 
    # If corrected estimate is outside stationary region, reduce bias correction little by little
    # (as recommended by Kilian & Lütkepohl, 2017, ch. 12)
    delta = 1
    while np.max(np.abs(np.linalg.eig(A_corr)[0])) > 1 and delta > 0:
        delta = delta - 0.01
        A_corr = A_comp + delta*b/T

    # Return bias-corrected VAR(p) coefficients
    A_corr = A_corr[:n,:]
  
    return A_corr



A_in = np.array([[1,2,3,2],[4,3,2,5]])
Sigma_in = np.array([[1,2],[2,1]])
T_in = 20

output = var_biascorr(A_in,Sigma_in,T_in)
print(output)