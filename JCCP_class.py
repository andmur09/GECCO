import numpy as np
from math import log, exp
from scipy.stats import multivariate_normal as norm
from additional_functions import *
import copy

class JCCP(object):
    def __init__(self, A, vars, b, c, T, q, mean, cov, alpha):
        self.A = A
        self.vars = vars
        self.b = b
        self.c = c
        self.T = T
        self.q = q
        self.mean = mean
        self.cov = cov
        self.alpha = alpha
        self.z = None
        self.phi = None
        self.duals = None
        self.cbasis = None
        self.probability = None

    def setZ(self, z):
        self.z = z
    
    def setPhi(self, phi):
        self.phi = phi

    def setProbability(self, phi):
        self.probability = exp(-phi)

    def getDuals(self):
        return copy.deepcopy(self.duals)

    def setDuals(self, duals):
        self.duals = duals
    
    def setCbasis(self, cbasis):
        self.cbasis = cbasis
    
    def addColumn(self, z_k):
        phi_k = self.calculatePhi(z_k)
        try:
            self.z = np.hstack((self.z, z_k))
            self.phi = np.append(self.phi, phi_k)
        except:
            raise AttributeError("Matrix z and vector phi not yet initialised")

    def getPi(self):
        return -log(1-self.alpha)
    
    def calculatePhi(self, z):
        return -log(prob(flatten(z), self.mean, self.cov))

    def reducedCost(self, z):
        return np.transpose(self.duals["u"])@z + self.duals["v"] *  self.calculatePhi(z) + self.duals["nu"]
    


    
    
