import argparse
import numpy as np
import pandas as pd 
from scipy.stats import invgauss

#_old function [irs, ses, cis_dm, cis_boot] = ir_estim(Y, p, horzs, varargin)

Y = np.array([[1,2,1,3,4,2,1,2,3,2,4,1], [4,4,5,3,2,4,5,6,7,2,3,2]])
print(Y)

p = 2
horz = 10


def ir_estim(Y, p, horz,
            resp_var = 1,
            innov = np.array([1]),
            estimator = 'lp',
            alpha = 0.05,
            lag_aug = False,
            bias_corr = True,
            se_homosk = False,
            no_const = False,
            har = 1, #issue
            har_bw = 1,
            har_cv = 1,
            bootstrap = 1,
            boot_num = 1000,
            boot_lag_aug = False,
            boot_workers = 1,
            verbose = False
                ):
           
           #assert resp_var != 0, "Invalid Operation" 


    # Inputs: see above
    
    # Outputs:
    # irs       1 x H       estimated impulse responses at select horizons
    # ses       1 x H       s.e. for impulse responses
    # cis_dm    2 x H       lower and upper limits of delta method confidence intervals
    # cis_boot  2 x H x 3   lower and upper limits of bootstrap confidence intervals (3rd index: type of interval, either Efron, Hall, or Hall percentile-t)
    
    
    ## Preliminaries
    
    [T,n] = Y.shape;              # Dimensions
    
    nh = horz # Number of horizons
    
    cvs = np.full(1,horz,invgauss((1-alpha)/2));      # Default critical values: normal
    #cvs = repmat(norminv((1-alpha)/2)),1, horz);      # Default critical values: normal

    cis_boot = np.array(2,nh,3);       # Initializes NaN array

    #First index is zero in numpy
    innov = innov - 1

    # Determine linear combination of innovations
    if innov.shape[1]==1:
        the_eye = np.identity(n)
        nu = the_eye(1,innov) # Unit vector
    else:
        nu = innov; # User-specified vector
    

    ## Point estimates and var-cov
    
    if estimator == "var": # VAR
        
        # VAR impulse responses
        [irs_all, irs_all_varcov] = var_ir_estim(Y, ...
                                                 p,...
                                                 p+ip.Results.lag_aug,...
                                                 horzs, ...
                                                 ip.Results.bias_corr,...
                                                 ip.Results.se_homosk,...
                                                 ip.Results.no_const);
        
        # Impulse responses of interest and s.e.
        [irs, ses] = var_select(irs_all, irs_all_varcov, resp_var, nu);
        
    elif estimator == "lp": # LP
        
        irs = np.zeros(1,nh);
        
        ses = np.zeros(1,nh);
        
        betahat = np.array(1,nh);
        
        res = np.array(1,nh);
        
        X   = np.array(1,nh);
        
        for h in range(nh) # For each horizon...
            
            the_horz = h+1; # Horizon
            
            # Determine s.e. setting and c.v.
            
            #ISSUE this block with all of the har stuff is being ignored for for
            if har ==555: # HAR ISSUE
                #the_bw = ip.Results.har_bw(T-p-ip.Results.lag_aug-the_horz); 
                    # Bandwidth, determined by effective sample size
                    
                #the_se_setting = @(X) ip.Results.har(X,the_bw); 
                    # HAR function
                 
                #if har_cv ==555:
                    cvs(h) = ip.Results.har_cv(the_bw); # Critical value
                
                
            else: # EHW/homoskedastic
                
                the_se_setting = se_homosk; # Indicator for whether homosk. or EHW
            
            
            
            # LP regression
            [the_irs_all, the_irs_all_varcov, betahat{h},...
            ~, res{h}, X{h}] = lp(Y, ...
                                  p-1+ip.Results.lag_aug,...                                                                                          
                                  the_horz,...      
                                  ip.Results.resp_var,...
                                  the_se_setting,...
                                  ip.Results.no_const);
            [irs(h),ses(h)] = lp_select(the_irs_all, the_irs_all_varcov, nu);
            
        
        
    
    # If only point estimates and standard errors are requested, stop
    # issue 
    #if nargout<=2
       # return;
    #end
    
    
    ## Delta method confidence intervals
    
    cis_dm = irs + [-1; 1]*(cvs.*ses);
    
    
    ## Bootstrap confidence intervals
# """     """  
#     if bootstrap == 1:
        
#         estims_boot = np.zeros(boot_num,nh);
#         ses_boot = estims_boot
        
#         if bootstrap == 'var': # Recursive VAR bootstrap specifications
            
#             # VAR coefficient estimates that define bootstrap DGP
            
#             [irs_var, ~, Ahat_var, res_var] = var_ir_estim(Y, ...
#                                                   p+ip.Results.boot_lag_aug,... # Compute impulse responses for full lag-augmented model, if applicable
#                                                   p+ip.Results.boot_lag_aug,...
#                                                   horzs, ...
#                                                   ip.Results.bias_corr,...
#                                                   ip.Results.se_homosk,...
#                                                   ip.Results.no_const);
            
#             pseudo_truth = var_select(irs_var, [], ip.Results.resp_var, nu); # Pseudo-true impulse responses in bootstrap DGP
            
# #             for b=1:ip.Results.boot_num
#             parfor(b=1:ip.Results.boot_num, ip.Results.boot_workers)

#                 # Generate bootstrap sample based on (possibly lag-augmented) VAR estimates
#                 Y_boot = var_boot(Ahat_var, res_var, Y, p+ip.Results.boot_lag_aug, ip.Results.se_homosk, ip.Results.no_const);

#                 # Estimate on bootstrap sample
#                 [estims_boot(b,:),ses_boot(b,:)] = ir_estim(Y_boot, p, horzs, varargin{:});
                
#                 # Print progress
#                 print_prog(b, ip.Results.boot_num, ip.Results.verbose);

#             end

#         else # Linear regression bootstrap specifications
            
#             pseudo_truth = irs;
            
# #             for b=1:ip.Results.boot_num
#             parfor(b=1:ip.Results.boot_num, ip.Results.boot_workers)
  
#                 for h=1:nh # Treat each horizon separately
                    
#                     # Generate bootstrap sample
#                     [Y_boot, X_boot] = linreg_boot(betahat{h}',res{h},X{h},ip.Results.bootstrap,ip.Results.se_homosk);

#                     # Run OLS on bootstrap sample
#                     [the_linreg_betahat, the_linreg_varcov] = linreg(Y_boot,X_boot,ip.Results.se_homosk,true); # Don't add extra intercept
#                     [estims_boot(b,h),ses_boot(b,h)] = lp_select(the_linreg_betahat(1:n),the_linreg_varcov(1:n,1:n),nu);

#                 end
                
#                 # Print progress
#                 print_prog(b, ip.Results.boot_num, ip.Results.verbose);

#             end
        
#         end
        
#         # Compute bootstrap confidence intervals
#         cis_boot = boot_ci(pseudo_truth, irs, ses, estims_boot, ses_boot, ip.Results.alpha);
        
#      """ """

#function [irs, ses] = var_select(irs_all, irs_all_varcov, resp_var, nu)
def var_select(irs_all, irs_all_varcov, resp_var, nu):

    # VAR: Return impulse responses of interest along with s.e.

    #irs = nu*permute(irs_all(resp_var,:,:), [2 3 1]);
    A = irs_all(resp_var,...,...)
    permute = np.transpose( np.expand_dims(A, axis=2), (1, 2, 0) )
    irs = nu*permute

    #irs = nu*permute(irs_all(resp_var,:,:), [2 3 1]);

    # Standard errors
    if len(*args)>3: #ISSUE HERE 
        [n,...,nh] = size(irs_all)
        the_eye = np.identity(n)
        #another function
        aux = kron(nu',the_eye(resp_var,:));
        ses = zeros(1,nh);
        for h=1:nh
            ses(h) = sqrt(aux*irs_all_varcov(:,:,h)*aux');
        end
    end



#function [ir, se] = lp_select(irs_all, irs_all_varcov, nu)
def lp_select(irs_all, irs_all_varcov, nu):
    # Local projection: Return impulse response of interest along with s.e.
    ir = irs_all*nu;
    se = sqrt(nu*irs_all_varcov*nu);
    return [ir, se]


# function print_prog(b, n, verbose)
   
#     # Print progress to screen periodicially
    
#     if verbose == True
#         return;
    
    
#     if mod(b,ceil(n/10))==0
#         fprintf('#3d#s\n', 100*b/n, '#');
#     end
    
# end