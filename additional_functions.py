import compute_probabilities as prob
import numpy as np
from math import sqrt, log

def reducedCost(z, u, v, nu, mu, cov):
    return np.transpose(u)@z + v *  -log(prob.pmvnorm(z, mu, cov)) + nu

def dual(z, u, v, mu, cov):
    return -np.transpose(u)@z - v * -log(prob.pmvnorm(z, mu, cov))

def gradDual(z, u, v, cb, mean, cov):
    return v/prob.pmvnorm(z, mean, cov)* prob.grad(z, cb, mean, cov) - u

def flatten(column_vector):
        return np.transpose(column_vector).flatten()