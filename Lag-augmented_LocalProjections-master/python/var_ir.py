from kron_fast import * 

def var_ir(A, horzs):

    # VAR(p) reduced-form impulse responses and Jacobian wrt. parameters
    
    # Inputs:
    # A         n x np  VAR coefficient matrices (A_1, ..., A_p)
    # horzs     1 x H   horizons of interest
    
    # Outputs:
    # irs       n x n x H           reduced-form impulse responses Theta_h at select horizons
    # jacob     n^2 x (n^2*p) x H   Jacobian of vec(Theta_h) at select horizons wrt. vec(A)
        
    nh = horzs.shape[1]
    maxh = np.max(horzs)
    (n, np) = A.shape
    p = np / n

    irs = np.zeros((n, n, nh)) # will contain impulse responses at select horizons
    jacob = np.zeros((n ** 2, n ** 2 * p, nh)) # will contain Jacobian of impulse responses at select horizons wrt vec(A)

    ir_p = np.vstack((np.eye(n), np.zeros((n * (p - 1), n)))) # will contain last p impulse responses, stacked vertically
    jacob_p = np.zeros((n ** 2, n ** 2 * p, p)) # will contain last p values of the Jacobian of vec(Theta_h) wrt vec(A)

    for h in range(1, maxh + 1): # loop through horizons
        the_A = A[:, :n * min(h, p)]
        the_past_ir = ir_p[:n * min(h, p), :]
        the_ir = the_A @ the_past_ir # impulse response at horizon h
        ir_p = np.vstack((the_ir, ir_p[:n, :])) # shift forward in time

        the_ind = next((ind for ind, x in enumerate(horzs) if x == h), None)
        if the_ind is not None:
            irs[:, :, the_ind] = the_ir # store impulse response at select horizons

        # Jacobian at horizon h via chain rule
        the_jacob_p = np.zeros((n ** 2, n ** 2 * p))
        the_jacob_p[:, :n ** 2 * min(h, p)] = np.kron(the_past_ir.T, np.eye(n))
        for l in range(1, min(h, p) + 1):
            the_jacob_p[:, :n ** 2 * min(h, p)] += kron_fast(A[:, (l - 1) * n:l * n], jacob_p[:, :n ** 2 * min(h, p), l], 1)

        # shift forward in time
        jacob_p[:, :,1:] = jacob_p[:,:,0:jacob_p.shape[2]-1]
        jacob_p[:,:,0] = the_jacob_p

        # if ~isempty(the_ind)
        #     jacob(:,:,the_ind) = the_jacob_p; % Store Jacobian at select horizons
        # end
     