from rpy2.robjects.packages import importr
from rpy2 import robjects as ro
from scipy.stats.mvn import mvnun as rectangular
import numpy as np
import time

# Using rpy2
base = importr('base')
print(base._libPaths())
mvtnorm = importr('mvtnorm', lib_loc = "/home/andmur09/R/x86_64-pc-linux-gnu-library/4.1")
lower = ro.IntVector((-1, -1))
upper = ro.IntVector((1, 1))
mean = ro.IntVector((0, 0))
cov = ro.r.matrix(ro.IntVector((1, 0, 0, 1)), nrow=2)
beginr = time.time()
print(mvtnorm.pmvnorm(lower, upper, mean, cov))
endr = time.time()

# Using Scipy
lower = np.array([-1, -1])
upper = np.array([1, 1])
mean = np.array([0, 0])
cov = np.array([[1, 0],[0, 1]])
beginp = time.time()
print(rectangular(lower, upper, mean, cov))
endp = time.time()

print("R time = ", endr - beginr, "Python time = ", endp - beginp)