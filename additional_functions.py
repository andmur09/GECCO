import compute_probabilities as prob
import numpy as np
from math import sqrt, log

def reducedCost(u, v, nu, z, mu, cov):
    return np.transpose(u)@z + v *  -log(prob.pmvnorm(z, mu, cov)) + nu

def dual(u, v, z, mu, cov):
    return -np.transpose(u)@z - v * -log(prob.pmvnorm(z, mu, cov))

def gradDual(u, v, z, cb, mean, cov):
    return v/prob.pmvnorm(z, mean, cov)* prob.grad(z, cb, mean, cov) - u