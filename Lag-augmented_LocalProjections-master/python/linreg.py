import numpy as np

from kron_fast import *
#THIS IS A UTILITY FOR TESTING
Y_in = np.array([1,2,1,3,4,2,1,2,3,2,4,1])
X_in = np.array([4,4,5,3,2,4,5,6,7,2,3,2])
Y_in = np.transpose(Y_in)
X_in = np.transpose(X_in)
######################################



def linreg(Y, X, se_setting, no_const):
    # Intputs:
    # Y         T x n   dependent variable data matrix
    # X         T x k   covariate data matrix
    # se_setting        EITHER bool: if true, homoskedastic s.e.; if false, EHW s.e.
    #                   OR function handle: function that returns HAC/HAR sandwich matrix
    # no_const  bool    true: omit intercept
    
    # Outputs:
    # betahat   n x (k+~no_const)       estimated coefficients
    # varcov    (n(k+~no_const)) x      var-cov of vec(betahat)
    #           (n(k+~no_const))
    # res       T x n                   residual matrix
    # X_expand  T x (k+~no_const)       covariate data matrix (if no_const)expanded covariate data matrix with intercept

    import numpy as np

    test = np.shape(Y)[0]
    print("this is the shape")
    print(test)
    
    T = np.shape(Y)[0]
    try:
        n = np.shape(Y)[1]
    except IndexError:
        n = 1
  

    print(n)


    # Include intercept if desired
    if no_const == True:
        X_expand = X
    else:
        temp = np.ones(T)
        #print(temp)
        print(X.shape)
        print(np.ones(T).shape)
        X_expand = np.column_stack([X, np.ones(T)])
        print(X_expand.shape)

    k = X_expand.shape[1]

    # OLS
    print(Y.shape)
    
    betahat = np.linalg.lstsq(X_expand, Y, rcond=None)[0]
    print(betahat)

    # Standard errors

    res = Y - np.dot(X_expand, betahat)
    XpX = np.dot(X_expand.T, X_expand)
    test1 = np.tile(res,(1,n))
    test1 = np.transpose(test)
    print(X_expand.shape)
    print(test1.shape)
    
    scores = np.kron(X_expand, np.ones((1, n))) * test1
    #scores = np.kron(X_expand, test)

    print("score matrix is below")
    #print(scores)
    #CLEAN THE CODE ABOVE

    if isinstance(se_setting, bool):
        if se_setting:  # If homoskedastic s.e.
            varcov = np.kron(np.linalg.inv(XpX), (res.T @ res) / T)
        else:  # If EHW s.e.
            
            varcov = kron_fast(np.linalg.inv(XpX), kron_fast(np.linalg.inv(XpX), scores.T @ scores,0).T,0).T  # EHW var-cov matrix
    else:  # If HAC/HAR s.e.
        varcov = kron_fast(np.linalg.inv(XpX), kron_fast(np.linalg.inv(XpX), se_setting(scores),0).T,0).T  # HAC/HAR var-cov matrix

    varcov = T / (T - k) * varcov  # Finite sample adjustment as in Stata

    print(varcov)
    print(betahat)
    return betahat, varcov, res, X_expand
    

test = linreg(Y_in,X_in,False,False)

#print(test)
