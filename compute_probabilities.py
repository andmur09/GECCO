
from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
from scipy.stats.mvn import mvnun as rectangular
import numpy as np
import time
    # mvtnorm = importr('mvtnorm', lib_loc = "/home/andmur09/R/x86_64-pc-linux-gnu-library/4.1")
    # upper = ro.IntVector((1, 1))
    # mean = ro.IntVector((0, 0))
    # cov = ro.r.matrix(ro.IntVector((1, 0, 0, 1)), nrow=2)
    # beginr = time.time()
    # print(mvtnorm.pmvnorm(upper=upper, mean=mean, sigma=cov))
    # endr = time.time()



def computeProbabilitySingular(z, mean, cov):
    mvtnorm = importr('mvtnorm', lib_loc = "/home/andmur09/R/x86_64-pc-linux-gnu-library/4.1")
    upper = ro.IntVector(z)
    mean = ro.IntVector(mean)
    cov = ro.r.matrix(ro.IntVector(cov.flatten('f')), nrow=np.shape(cov)[0])
    result = mvtnorm.pmvnorm(upper=upper, mean=mean, sigma=cov)
    return np.asarray(result)[0]

print(computeProbabilitySingular(np.array([1, 1]), np.array([0, 0]), np.array([[1, 0], [0, 1]])))