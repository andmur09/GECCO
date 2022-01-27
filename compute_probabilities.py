
from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
import numpy as np
from scipy.stats import multivariate_normal as norm

def pmvnorm(z, mean, cov):
    mvtnorm = importr('mvtnorm', lib_loc = "/home/andmur09/R/x86_64-pc-linux-gnu-library/4.1")
    upper = ro.IntVector(z)
    mean = ro.IntVector(mean)
    cov = ro.r.matrix(ro.IntVector(cov.flatten('f')), nrow=np.shape(cov)[0])
    result = mvtnorm.pmvnorm(upper=upper, mean=mean, corr=cov)
    return np.asarray(result)[0]

def grad(z, u, mean, cov):
    n = int(np.shape(mean)[0])
    dz = []
    for i in range(n):
        if u[i] == 0:
            dz.append(0)
        else:
            bar_mean = np.delete(mean, i)
            bar_cov = np.delete(np.delete(cov, i, 0), i, 1)
            bar_z= np.delete(z, i)
            bar_F = pmvnorm(bar_z, bar_mean, bar_cov)
            f = norm(mean[i], cov[i, i]).pdf(z[i])
            dz.append(f * bar_F)
    return np.c_[np.array(dz)]

#print("x")
#print(np.array(2, 3))