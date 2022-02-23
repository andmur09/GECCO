
def columnGeneration(z, JCCP, iterations = 10, epsilon = 0.001, lb = False):
    '''
    Description:    Solves the column generaion problem (below) via gradient descent with backtracking line search:

                    min_z.  -u^Tz - v*phi(z) - nu    

                    And returns a column z which optimises the reduced cost.
    
    Input:          JCCP:   Instance of JCCP class
    Output:         m:      An instance of the Gurobi model class
    '''
    duals = JCCP.getDuals()
    u, v, nu = fn.flatten(duals["u"]), duals["v"], duals["nu"]
    mean, cov = JCCP.mean, JCCP.cov
    cb = JCCP.cbasis
    z = fn.flatten(z)
    start = time.time()
    def dualf(z):
        # Nested function to be optimised
        return -np.dot(u, z) - v * -log(max(norm(mean, cov, allow_singular=True).cdf(z), sys.float_info.min))- nu

    def gradf(z):
        # Nested function to calculate gradients at particular points
        return fn.flatten(v/max(norm(mean, cov, allow_singular=True).cdf(z), sys.float_info.min) * fn.grad(np.c_[z], cb, mean, cov)) - u

    def backtracking(z, grad, beta=0.8, alpha = 0.1):
        # Implementation of backtracking line search using Armijos condition
        t = 1
        try:
            dualf(z-t*grad)
        except ValueError:
            t *= beta
        print("\nBACKTRACKING BEGINS HERE")
        #print("LHS = ", dualf(z-t*grad), "RHS = ", dualf(z) - alpha * t * np.dot(gradf(z), gradf(z)), "t = ", t)
        while dualf(z-t*grad) > dual - alpha * t * np.dot(gradf(z), gradf(z)):
            t *= beta
            #print("LHS = ", dualf(z-t*grad), "RHS = ", dualf(z) - alpha * t * np.dot(gradf(z), gradf(z)), "t = ", t)
        print("BACKTRACKING ENDS HERE")
        return t
    
    # Initialises values for the function and gradient.
    dual, grad = dualf(z), gradf(z)
    print("\nITERATION NUMBER: 0")
    print("z = ", z)
    #print("Prob = ", fn.prob(z, mean, cov))
    print("Grad = ", grad)
    print("Function value = ", dual)
    print("Reduced cost = ", -dual)
    # Calculates step size using backtracking lnie search and performs gradient descent iterations until the number of iterations limit
    # is reached or the gradient is within an allowable tolerance.
    for i in range(1,iterations+1):
        t = backtracking(z, grad)
        z = z - t * grad
        dual, grad = dualf(z), gradf(z)
        print("\nITERATION NUMBER: {}".format(i))
        print("z = ", z)
        #print("Prob = ", norm(mean, cov, allow_singular=True).cdf(z))
        print("Grad = ", grad)
        print("Function value = ", dual)
        print("Reduced cost = ", -dual)
        print("Grad^2 = ", grad.transpose() @ grad)
        if lb == False and -dual > 0:
            end = time.time()
            print("Time taken: ", end - start)
            return np.c_[z]
        elif grad.transpose() @ grad < epsilon:
            return np.c_[z]
    print("Maximum number of iterations reached")
    return np.c_[z]