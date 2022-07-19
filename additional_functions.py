import numpy as np
#from rpy2.robjects.packages import importr
#from rpy2 import robjects as ro
import numpy as np
from scipy.stats import multivariate_normal as norm
from scipy import stats
from math import sqrt
import sys
from gurobipy import GRB
import gurobipy as gp
#from sqlalchemy import true

def flatten(column_vector):
    '''
    Description:    Changes numpy column vector to array    
    Input:          A numpy P x 1 column vector
    Output:         A numpy p dimensional array 
    '''
    return np.transpose(column_vector).flatten()

#def pmvnorm(z, mean, cov):
 #   '''
  #  Description:    Uses algorithm from: Genz, A. and Kwong, K.S., 2000. Numerical evaluation of singular multivariate normal distributions
   #                 to calculate probability F(z) of singular multivariate normal distributions
    #Input:          z:      a vector at which to evaluate the probability
   #                 mean:   a vector of mean values
   #                 cov:    a covariance matrix
   # Output          float:  Value of F(z) for N(mean, cov)
   # '''
   # mvtnorm = importr('mvtnorm', lib_loc = "/home/andmur09/R/x86_64-pc-linux-gnu-library/4.1")
   # upper = ro.FloatVector(z)
   # mean = ro.FloatVector(mean)
   # cov = ro.r.matrix(ro.FloatVector(cov.flatten('f')), nrow=np.shape(cov)[0])
   # result = mvtnorm.pmvnorm(upper=upper, mean=mean, sigma=cov)
   # return np.asarray(result)[0]

def prob(z, mean, cov):
    '''
    Description:    Scipy multivariate normal cdf to evaluate the probablity F(z)
    Input:          z:      a vector at which to evaluate the probability
                    mean:   a vector of mean values
                    cov:    a covariance matrix
    Output          float:  Value of F(z) for N(mean, cov)
    '''
    prob = norm(mean, cov, allow_singular=True).cdf(z)
    if prob == 0:
        return sys.float_info.min
    else:
        return prob


def grad(z, mean, cov, psi):
    '''
    Description:    Calculates gradient vector of probability of singular multivariate normal distributions grad(F(z)) based on:
                    Henrion, R. and Möller, A., 2012. A gradient formula for linear chance constraints under Gaussian distribution.
    Input:          z:      a vector at which to evaluate the gradient
                    cb:     Gurobi cbasis vector used to determine nondegeneracy indices
                    mean:   a vector of mean values
                    cov:    a covariance matrix
    Output          numpy:  Column vector of partial derivatives of grad(F(z)). Each index is partial(F(z))/z_i
    '''
    n = int(np.shape(mean)[0])
    I = get_active_indices(z, mean, psi)
    dz = []
    for i in range(len(I)):
        if  I[i] != 0:
            dz.append(0)
        else:
            bar_mean = np.delete(mean, i)
            bar_cov = np.delete(np.delete(cov, i, 0), i, 1)
            bar_z= np.delete(z, i)
            bar_F = norm(bar_mean, bar_cov, allow_singular=True).cdf(bar_z)
            f = norm(mean[i], sqrt(cov[i, i])).pdf(z[i])
            dz.append(f * bar_F)
    return np.c_[np.array(dz)]

def get_active_indices(z, mean, psi):
    shape = np.shape(psi)
    z = flatten(z)
    print("Z: ", z)
    m, s = shape[0], shape[1]
    I = []
    for i in range(m):
        # Sets up and solves the LP from Henrion and Moller 2012 (proposition 4.1)
        model = gp.Model("index_check_{}".format(i))
        model.setParam('OutputFlag', 0)
        u = model.addMVar(m)
        x = model.addMVar(s)
        model.addConstr(psi @ x + u == z - mean)
        model.setObjective(u[i], GRB.MINIMIZE)
        model.update()
        model.optimize()
        # Checks to see whether an optimal solution is found and if so it prints the solution and objective value
        if model.status != GRB.OPTIMAL:
            model.computeIIS()
            model.write("gurobi_files/active_indices.lp")
            model.write("gurobi_files/active_indices.ilp")
        I.append(model.objVal)
    print("I = ", I)
    return I

def generate_random_correlation(n, eta, size=1):
    beta0 = eta - 1 + n/2
    shape = n * (n-1) // 2
    triu_ind = np.triu_indices(n, 1)
    beta_ = np.array([beta0 - k/2 for k in triu_ind[0]])
    # partial correlations sampled from beta dist.
    P = np.ones((n, n) + (size,))
    P[triu_ind] = stats.beta.rvs(a=beta_, b=beta_, size=(size,) + (shape,)).T
    # scale partial correlation matrix to [-1, 1]
    P = (P-.5)*2
    
    for k, i in zip(triu_ind[0], triu_ind[1]):
        p = P[k, i]
        for l in range(k-1, -1, -1):  # convert partial correlation to raw correlation
            p = p * np.sqrt((1 - P[l, i]**2) *
                            (1 - P[l, k]**2)) + P[l, i] * P[l, k]
        P[k, i] = p
        P[i, k] = p

    return np.transpose(P, (2, 0 ,1))[0]