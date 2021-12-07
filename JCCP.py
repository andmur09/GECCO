import convex
from scipy.stats import multivariate_normal
from math import log
import numpy as np
from scipy.stats.mvn import mvnun
import numpy as np

def check_initial(z0, delta, omega):
    probability = omega.cdf(z0[1]) - omega.cdf(z0[0])
    if probability >= 1 - delta:
        return True
    else:
        return False

def JCCP(PSTN, alpha):
    #Assign variables
    k = 0
    z = np.array([])

    # Get Matrices
    mat = convex.get_matrices(PSTN)
    A, vars, b, c, T_l, T_u, q_l, q_u, mu, cov = mat[0], mat[1], mat[2], mat[3], mat[4], mat[5], mat[6], mat[7], mat[8], mat[9]
    omega = multivariate_normal(mu, cov)

    #Calculates initial z and checks whether risk bound is satisfied
    m = convex.get_initial_points(A, vars, b, T_l, T_u, q_l, q_u)
    results[str(k)]["model"] = m[0]
    np.append(z, m[1])
    if check_initial(z[0], alpha, mu, cov) == False:
        return False
    else:
        np.append(phi, -omega.logcdf(z[0]))

    