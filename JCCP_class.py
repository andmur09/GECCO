import numpy as np
from math import log, exp
from scipy.stats import multivariate_normal as norm
from additional_functions import *
import copy

class JCCP(object):
    """
    Description:    Class representing a Joint Chance Constrained Problem
    Parameters:
                A:              numpy m x n matrix of coefficients
                vars:           numpy n dimensional decision vector
                b:              numpy m dimensional vector of RHS values
                c:              numpy n dimensional vector of objective coefficients
                T:              numpy p x n matrix of coefficients
                q:              numpy p dimensional vector of RHS values
                mu:             numpy p dimensional mean vector of 
                cov:            numpy p x p dimensional covariance matrix
                alpha:          float allowable tolerance on risk such that P(success) >= 1-alpha and alpha \in {0, 1}
                z:              numpy p x k matrix of current approximation points
                phi:            numpy k dimensional vector of phi(z) = -log(F(z)) for approximation point (columns in z)
                duals:          dictionary {"u": numpy array, "v": float, "nu": float, "mu": numpy array} of dual variables for the
                                current iteration. The dual variables are associated with the following constraints:
                                u: Z @ lambda - T @ x <= q, v: phi^T @ lambda, nu: 1^T @ lambda = 1, mu: A @ x <= b
                cbasis:         numpy Gurobi attribute class associated with current master problem: 
                                https://www.gurobi.com/documentation/9.5/refman/cbasis.html
                solved:         bool True if solution is optimal else False
                solution:       dictionary {variable name: value,..,objective: objective value}
                solution_time:  float time taken to reach solution
    """
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
        self.solved = False
        self.solution = []
        self.solution_time = None
        self.start_i = None
        self.convergence_time = []
        self.master_time = []

    def setZ(self, z):
        # Sets z during initialisation
        self.z = z
    
    def setPhi(self, phi):
        # Sets phi during initialisation
        self.phi = phi

    def getDuals(self):
        # Returns a copy of the dual variable dictionary
        return copy.deepcopy(self.duals)
    
    def getSolution(self):
        return self.solution

    def setDuals(self, duals):
        # Sets duals based on current solution to master problem
        self.duals = duals
    
    def setCbasis(self, cbasis):
        # Sets duals based on current solution to master problem
        self.cbasis = cbasis
    
    def addColumn(self, z_k, phi_k):
        # Adds a column z_k to matrix z and item phi_k to vector of phi values
        try:
            self.z = np.hstack((self.z, z_k))
            self.phi = np.append(self.phi, phi_k)
        except:
            raise AttributeError("Matrix z and vector phi not yet initialised")

    def getPi(self):
        # Returns value of -log(1-alpha) to be used in joint chance constraint
        return -log(1-self.alpha)
    
    def calculatePhi(self, z):
        # Calculates value of -log(F(z)) for a column z
        return -log(prob(flatten(z), self.mean, self.cov))

    def reducedCost(self, z):
        # Calculates reduced cost using current dual variables and column
        return np.transpose(self.duals["u"])@z + self.duals["v"] *  self.calculatePhi(z) + self.duals["nu"]
    
    def setSolved(self, status):
        self.status = status
    
    # Takes a Gurobi model and adds a solution containing variable and objective values.
    def addSolution(self, model):
        solution = {}
        for v in model.getVars():
            solution[v.varName] = v.x
        solution["Objective"] = model.objVal
        self.solution = solution
    
    def setSolutionTime(self, time):
        self.solution_time = time
    
    def getCurrentProbability(self):
        #print(self.solution.keys())
        for key in self.solution.keys():
            if "phi" in key:
                return exp(-self.solution[key])
    
    def add_convergence_time(self, time, gap):
        self.convergence_time.append((time, gap))
    
    def add_master_time(self, time, cost):
        self.master_time.append((time, cost))
    


    
    
