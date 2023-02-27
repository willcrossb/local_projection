#This defines a simple local projection function in python
#Dependencies
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

"""
DEFINE: locproj() => a function to estimate local projects as in Jorda 2005
-----------------------INPUTS -------------------------------
X: matrix containig the shock of interest +  controls
Y: response variable of interest (must be a single vector)
innov_idx: column index of the shock vector in X (indexes start at zero)
horizon: the # of periods you want to know the response for
sig_level: add as an # in (00,100): ex. 90 => function will output 90% confidence interval

------------------------OUTPUTS -----------------------------
irf: the dataframe of resulting response, confidence interval, and horizon
info_store: DF containing series info (units, titles, notes, etc.)
"""

def locproj(X,Y,innov_idx,horizon,sig_level):
    
    #Create an empty array to store results
    irf = np.empty([horizon+1,4])
 
    if Y.ndim > 1:
        raise Exception("INPUT ERROR: Y must be a single column vector")

    if not 0 < sig_level < 100:
        raise Exception("INPUT ERROR: significance level not in range 1 - 100")
    
    #loop from 0 to h periods ahead
    for h in range(0,horizon+1):
        X = sm.add_constant(X)
        #At each point, shift the Y matrix forward, while trimming X to keep them the same length
        X_reg = X[0:len(X)-h,:]
        Y_reg = Y[h:]
        model = sm.OLS(Y_reg, X_reg).fit()
        
        #Return the level for the confidence bands 
        sig_level_se = norm.ppf(1-((100-sig_level)/200))
        #store results in row h
        irf[h,0] = h
        irf[h,1] = model.params[innov_idx+1]
        
        #Create confidence interval bands
        irf[h,2] = irf[h,1]+(model.HC0_se[innov_idx+1]*(-1*sig_level_se))
        irf[h,3] = irf[h,1]+(model.HC0_se[innov_idx+1]*sig_level_se)
    
    #Collect resul matrix into a dataframe
    irf_df = pd.DataFrame(irf, columns = ['horizon','resp','se_low','se_high'])
    return irf_df