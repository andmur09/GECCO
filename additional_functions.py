import compute_probabilities as prob
import numpy as np
from math import sqrt, log

def reducedCost(z, u, v, nu, mu, cov):
    return np.transpose(u)@z + v *  -log(prob.pmvnorm(z, mu, cov)) + nu

def flatten(column_vector):
        return np.transpose(column_vector).flatten()