
import numpy as np
import os
import gurobipy as gp
from gurobipy import GRB
import sys
import numpy as np
import pickle as pkl
from scipy import stats

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

def getStandardForm(PSTN, name, pres = 15):
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

    controllables = PSTN.getControllables()
    requirements = PSTN.getRequirements()
    m.addVar(vtype=GRB.CONTINUOUS, name = "Risk")

    for i in controllables:
        m.addVar(vtype=GRB.CONTINUOUS, name=i.id)

    for requirement in requirements:
        if PSTN.isControllable(requirement) == True:
            m.addVar(vtype=GRB.CONTINUOUS, obj = requirement.intervals["value"], name = requirement.name + "_Rl")
            m.addVar(vtype=GRB.CONTINUOUS, obj = requirement.intervals["value"], name = requirement.name + "_Ru")
        else:
            incoming = PSTN.incomingContingent(requirement)
            if incoming["start"] and incoming["end"]:
                raise ValueError("Both start and end time-point are uncontrollable")
            elif incoming["start"] != None:
                incoming = incoming["start"]
            else:
                incoming = incoming["end"]
            m.update()
            if incoming.name + "_l" not in [v.varName for v in m.getVars()]:
                m.addVar(lb=0, ub = incoming.mu, vtype=GRB.CONTINUOUS, name = incoming.name + "_l")
                m.update()
            if incoming.name + "_u" not in [v.varName for v in m.getVars()]:
                m.addVar(lb = incoming.mu, ub = inf, vtype=GRB.CONTINUOUS, name = incoming.name + "_u")
                m.update()

            m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name = requirement.name + "_Fl")
            m.addVar(lb=0, ub=inf, vtype=GRB.CONTINUOUS, name = requirement.name + "_Fu")
            m.update()

    for constraint in requirements:
        # For controllable constraints with both start and end time-point is controllable
        if PSTN.isControllable(constraint) == True:
            m.update()
            # Collects indices of required variables in variable vector x
            start, end = m.getVarByName(constraint.source.id), m.getVarByName(constraint.sink.id)
            relax_l, relax_u = m.getVarByName(constraint.name + "_Rl"), m.getVarByName(constraint.name + "_Ru")
                    
            # Adds A matrix coeffcients for upper bound constraints of the form b_j - b_i - Gamma_u_{ij} * r_u_{ij} <= y_{ij}
            # Note sum has been moved to LHS of inequality, RHS of inequality "b" = 0
            m.addConstr(end - start - relax_u <=constraint.intervals["ub"])

            # Adds A matrix coefficients for lower bound constraints of the form b_i - b_j - Gamma_l_{ij} * r_l_{ij} <= -x_{ij}
            # Note sum has been moved to LHS of inequality, RHS of inequality "b" = 0
            m.addConstr(end - start + relax_l >= constraint.intervals["lb"])

        # For uncontrollable constraints containing an incoming contingent
        else:
            incoming = PSTN.incomingContingent(constraint)
            ## Start time-point in constraint is uncontrollable
            if incoming["start"] != None:
                incoming = incoming["start"]
                m.update()
                start, end = m.getVarByName(incoming.source.id), m.getVarByName(constraint.sink.id)

                if incoming.type == "pstc":
                    omega_l, omega_u = m.getVarByName(incoming.name + "_l"), m.getVarByName(incoming.name + "_u")
                    F_l, F_u = m.getVarByName(constraint.name + "_Fl"), m.getVarByName(constraint.name + "_Fu")

                    # For constraint of the form bj - bi - l_i <= y_{ij}
                    m.addConstr(end - start - omega_l <= constraint.intervals["ub"])
                    # For constraint of the form bi - bj + u_i <= -x_{ij}
                    m.addConstr(end - start - omega_u >= constraint.intervals["lb"])

                    # If incoming is probabilistic and distribution of incoming is Gaussian
                    if incoming.dtype() == "Gaussian":
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
                m.update()
                start, end = m.getVarByName(constraint.source.id), m.getVarByName(incoming.source.id)
                    
                if incoming.type == "pstc":
                    omega_l, omega_u = m.getVarByName(incoming.name + "_l"), m.getVarByName(incoming.name + "_u")
                    F_l, F_u = m.getVarByName(constraint.name + "_Fl"), m.getVarByName(constraint.name + "_Fu")

                    # For constraint of the form b_j + u_{ij} - b_i <= y_{ij}      
                    m.addConstr(end - start + omega_u <= constraint.intervals["ub"])        
                    # For constraint of the form b_i - bj - l_{ij} <= -x_{ij}
                    m.addConstr(end - start + omega_l >= constraint.intervals["lb"])

                    # If incoming is probabilistic and distribution of incoming is Gaussian
                    if incoming.dtype() == "Gaussian":
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
    m.update()
    risk = m.getVarByName("Risk")
    m.addConstr(gp.quicksum([v for v in m.getVars() if v.varName[-2:] in ["Fu", "Fl"]]) == risk, 'risk')

    m.addConstr(risk <= budget, "risk_bound")
    m.update()
    if log == True:
        m.write("{}/{}.mps".format(folder, m.ModelName))
    return m

def solveLP(PSTN):
    m = getStandardForm(PSTN)
    pass

path = "pstns/problems/woodworking/p01"

with open( path, "rb") as f:
    problem = pkl.load(f)
    solveLP(problem)
        