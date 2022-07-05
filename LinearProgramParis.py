
import numpy as np
import os
import gurobipy as gp
from gurobipy import GRB
import sys
import numpy as np
import pickle as pkl
from scipy import stats
from math import inf
import time

def linearProbability(constraint, m, n):
    '''
    Description: Returns piecewise linear points representing pdf.
    Input:  constraint - Instance of probabilistic constraint.
            m - Number of points to partition region to the left of the mode of the PDF
            n - Number of points to partition region to the right of the mode of the PDF
    Output: partitions_l = gradient, intercept pairs representing the piecewise linear segments to the left of the mode
            partitions_u = gradient, intercept pairs representing the piecewise linear segments to the right of the mode
    '''
    if constraint.dtype() == "Gaussian":
        mean = constraint.mu
        sd = constraint.sigma
        norm = stats.norm(mean, sd)

        # Since the PDF of omega is unbounded, in order to seperate the probability mass at set points, we must pick some arbitrarily large upper and arbitrarily small
        # lower bound. 
        cdf = 0.001
        lb = norm.ppf(cdf)
        ub = norm.ppf(1 - cdf)

        # Splits LHS of mean into m segments.
        LHS = np.linspace(lb, mean, m)
    
        # Calculates constants and gradients for each partition in LHS of mean
        partitions_l = []
        for i in range(m - 1):
            constant = 0
            for k in range(i):
                constant += norm.pdf(LHS[k+1])*(LHS[k+1] - LHS[k])
            m, c = norm.pdf(LHS[i+1]), cdf + constant - norm.pdf(LHS[i+1])*LHS[i]
            partitions_l.append([m, c])
        
        # Splits RHS of mean into n segments.
        RHS = np.linspace(mean, ub, n)

        # Calculates constants and gradients for each partition in RHS of mean
        partitions_u = []
        for j in range(0, n-1):
            constant = 0
            for k in range(j+1, n-1):
                constant += norm.pdf(RHS[k])*(RHS[k+1] - RHS[k])
            m, c = -norm.pdf(RHS[j]), cdf + constant + norm.pdf(RHS[j])*RHS[j+1]
            partitions_u.append([m, c])
        
    return (partitions_l, partitions_u)

def solveLP(PSTN, name, pres = 15):
    '''
    Description:    Makes matrices in the form Ax <= b and c^Tx representing the PSTN, which are used as input to the optimisation solver. For strong controllability Linear program.
    Input:          PSTN - Instance of PSTN to be solved
                    name - name to call model
                    pres - number of points for which to partition the function for probabilistic constraints (if pres = 50, then LHS of mode partitioned at 50 points, and RHS of mode partitioned at 50 points)
                    folder - folder to save Gurobi files to if log=True
                    log - if true writes Gurobi files to file
                    weight - weight to apply to relaxation cost terms in objective
    Output:         m - A Gurobi model containing all variables, constraints and objectives
    '''
    m = gp.Model(name)

    # Gets relevant items from PSTN
    cc = PSTN.getControllableConstraints()
    cu = PSTN.getUncontrollableConstraints()
    cp = PSTN.getProbabilisticConstraints()
    tc = PSTN.getControllables()
    
    #Adds Problem Variables
    m.addVar(vtype=GRB.CONTINUOUS, name = "Risk")

    for t in tc:
        m.addVar(vtype=GRB.CONTINUOUS, name=t.id)
    
    for c in cu:
        m.addVar(vtype=GRB.CONTINUOUS, obj = 1, name = c.name + "_Fl")
        m.addVar(vtype=GRB.CONTINUOUS, obj = 1, name = c.name + "_Fu")

    for c in cp:
        m.addVar(lb=0, ub=c.mu, vtype=GRB.CONTINUOUS, name = c.name + "_l")
        m.addVar(lb=c.mu, ub=inf, vtype=GRB.CONTINUOUS, name = c.name + "_u")
    m.update()

    # Adds constraints
    for c in cc:
        # Collects indices of required variables in variable vector x
        start, end = m.getVarByName(c.source.id), m.getVarByName(c.sink.id)
        # Adds constraint of the form b_j - b_i - r_u_{ij} <= y_{ij}
        m.addConstr(end - start <= c.intervals["ub"])
        # Adds constraint of the form b_i - b_j -  r_l_{ij} <= -x_{ij}
        m.addConstr(end - start >= c.intervals["lb"])
                    
    for c in cu:
        incoming = PSTN.incomingContingent(c)
        ## Start time-point in constraint is uncontrollable
        if incoming["start"] != None:
            incoming = incoming["start"]
            start, end = m.getVarByName(incoming.source.id), m.getVarByName(c.sink.id)
            omega_l, omega_u = m.getVarByName(incoming.name + "_l"), m.getVarByName(incoming.name + "_u")
            F_l, F_u = m.getVarByName(c.name + "_Fl"), m.getVarByName(c.name + "_Fu")
         
            # For constraint of the form bj - bi - l_i <= y_{ij}
            m.addConstr(end - start - omega_l <= c.intervals["ub"])
            # For constraint of the form bi - bj + u_i <= -x_{ij}
            m.addConstr(end - start - omega_u >= c.intervals["lb"])

             # Adds piecewise linear constraints.
            partitions = linearProbability(incoming, pres, pres)
            partitions_l, partitions_u = partitions[0], partitions[1]

            # Adds constraints of the form F_l >= grad * l + intercept
            for partition in partitions_l:
                grad, const = partition[0], partition[1]
                m.addConstr(F_l - grad*omega_l >= const)

            for partition in partitions_u:
                grad, const = partition[0], partition[1]
                m.addConstr(F_u - grad*omega_u >= const)
    
        ## End time-point in constraint is uncontrollable
        elif incoming["end"] != None:
            incoming = incoming["end"]
            start, end = m.getVarByName(c.source.id), m.getVarByName(incoming.source.id)
            omega_l, omega_u = m.getVarByName(incoming.name + "_l"), m.getVarByName(incoming.name + "_u")
            F_l, F_u = m.getVarByName(c.name + "_Fl"), m.getVarByName(c.name + "_Fu")
            
            # For constraint of the form b_j + u_{ij} - b_i <= y_{ij}      
            m.addConstr(end - start + omega_u <= c.intervals["ub"])        
            # For constraint of the form b_i - bj - l_{ij} <= -x_{ij}
            m.addConstr(end - start + omega_l >= c.intervals["lb"])

            # Adds piecewise linear constraints.
            partitions = linearProbability(incoming, pres, pres)
            partitions_l, partitions_u = partitions[0], partitions[1]

            # Adds constraints of the form F_l >= grad * l + intercept
            for partition in partitions_l:
                grad, const = partition[0], partition[1]
                m.addConstr(F_l - grad*omega_l >= const)

            for partition in partitions_u:
                grad, const = partition[0], partition[1]
                m.addConstr(F_u - grad*omega_u >= const)
                
    m.addConstr(m.getVarByName(PSTN.getStartTimepointName()) == 0)
    m.update()
    risk = m.getVarByName("Risk")
    m.addConstr(gp.quicksum([v for v in m.getVars() if v.varName[-2:] in ["Fu", "Fl"]]) == risk, 'risk')
    
    m.update()
    m.write("gurobi_files/{}.lp".format(PSTN.name))
    start = time.time()
    m.optimize()
    end = time.time()

    # Checks to see whether an optimal solution is found and if so it prints the solution and objective value
    results = {}
    results["Runtime"] = end - start
    if m.status == GRB.OPTIMAL:
        results["Objective"] = m.objVal
        print('\n objective: ', m.objVal)
        print('\n Vars:')
        for v in m.getVars():
            results[v.varName] = v.x
            print("Variable {}: ".format(v.varName) + str(v.x))
    else:
        m.computeIIS()
        m.write("gurobi_files/{}.ilp".format(PSTN.name))
    return m, results
        
