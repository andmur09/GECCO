import numpy as np
#from rpy2.robjects.packages import importr
#from rpy2 import robjects as ro
import numpy as np
from scipy.stats import multivariate_normal as norm
from math import sqrt, log
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
    return norm(mean, cov, allow_singular=True).cdf(z)

def grad(z, cb, mean, cov):
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
    dz = []
    for i in range(n):
        if cb[i] != -1:
            dz.append(0)
        else:
            bar_mean = np.delete(mean, i)
            bar_cov = np.delete(np.delete(cov, i, 0), i, 1)
            bar_z= np.delete(z, i)
            bar_F = norm(bar_mean, bar_cov, allow_singular=True).cdf(bar_z)
            f = norm(mean[i], sqrt(cov[i, i])).pdf(z[i])
            dz.append(f * bar_F)
    return np.c_[np.array(dz)]