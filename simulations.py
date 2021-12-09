# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 23:06:10 2021

@author: kpb20194
"""
import gurobipy as gp
from scipy.stats.stats import weightedtau
from scipy.stats import multivariate_normal
import LinearProgram as LP
import pickle as pkl
from gurobipy import GRB
import os
import numpy as np
import sys
import convex
inf = 10000

gurobi_status = {1: "Loaded", 2: "Optimal", 3: "Infeasible", 4: "Inf or unb", 5: "Unbounded", 6: "Cutoff", 7: "Iteration Limit", 8: "Node Limit", 9: "Time Limit", 10: "Solution Limit", 11: "Interrupted", 12: "Numeric", 13: "Suboptimal", 14: "In Progress", 15: "User Obj Limit"}

def solve(PSTN, folder, log = False, budget = inf):
    '''
    Description: Wrapper function to handle setup and solving of gurobi model as well as exceptions.
    Input:  PSTN - instance to be solved
            folder - folder to output results to
            log - if True saves gurobi solutions and troubleshooting to files
            weight - Adds a weight to the relaxation cost part of the objective.
    Output: None - returns none if error encountered
            result - gurobi model containing solution
    '''
    print("Getting matrices for instance: ", PSTN.name)
    
    try:
        m = LP.getMatricesLP(PSTN, PSTN.name, budget = budget, folder = folder, log = log)
    except AttributeError:
        print("Likely more than one incoming probabilistic link, could not solve")
        return None
    if m == None:
        print("Unable to solve PSTN")
        return None
    try:
        m = LP.LPSolve(m, folder=None, log = False)
        if m.status != GRB.OPTIMAL:
            return m
        m.remove(m.getConstrByName("risk_bound"))
        cost = m.addVar(vtype=GRB.CONTINUOUS, name = "Cost")
        m.addConstr(cost == m.objVal)
        m.addConstr(gp.quicksum([v for v in m.getVars() if v.varName[-2:] in ["Ru", "Rl"]]) == cost, 'cost')
        m.setObjective(m.getVarByName("Risk"))
        m.update()
        m = LP.LPSolve(m, folder = folder, log = log)
        return m
    except gp.GurobiError:
        print("Unable to add constraints, LP setup failed")
        return None

def getSchedule(PSTN, solution):
    '''
    Description: takes PSTN and gurobi solution and extracts schedule in form of dictionary of timepoint: value pairs.
    Input:  PSTN - instance to be solved
            solution - gurobi model containing solution
    Output: None - returns none if error encountered
            schedule - dictionary of timepoint, value pairs containing schedule
    '''
    try:
        variables = [v.varName for v in solution.getVars()]
        values = [v.x for v in solution.getVars()]
        timePoints = [t.id for t in PSTN.getTimePoints()]
        schedule = {}
        for i in range(len(variables)):
            if variables[i] in timePoints:
                schedule[variables[i]] = values[i]
        return schedule
    except:
        return None

def getRelaxations(PSTN, solution):
    try:
        variables = [v.varName for v in solution.getVars()]
        values = [v.x for v in solution.getVars()]
        controllables = [c.name for c in PSTN.getRequirements() if PSTN.isControllable(c) == True]
        relaxations = {}
        for c in controllables:
            relaxations[c] = {"Rl": None, "Ru": None}
        for i in range(len(variables)):
            if variables[i][-2:] == "Rl":
                relaxations[variables[i][:-3]]["Rl"] = solution.getVarByName(variables[i]).x
            elif variables[i][-2:] == "Ru":
                relaxations[variables[i][:-3]]["Ru"] = solution.getVarByName(variables[i]).x
        return relaxations
    except:
        return None

def main():
    epsilon = 0.01
    woodworking_path = "pstns/problems/woodworking"
    woodworking_files = sorted(os.listdir(woodworking_path))
    woodworking = []
    for i in range(1):
        with open(woodworking_path + "/" + woodworking_files[i], "rb") as f:
            problem = pkl.load(f)
            woodworking.append(problem)

    # elevators_path = "pstns/problems/elevators"
    # elevators_files = sorted(os.listdir(elevators_path))
    # elevators = []
    # for i in range(1):
    #     with open(elevators_path + "/" + elevators_files[i], "rb") as f:
    #         problem = pkl.load(f)
    #         elevators.append(problem)

    for instance in woodworking:
        # instance.plot()
        mat = convex.JCCP(instance, 0.4, 0.01)

        # upper = instance.countUncontrollables()
        # result = epsilonConstraint(instance, "pstns/results/woodworking", upper, epsilon, log = True)
        # print(result)
        # with open("pstns/results/woodworking/{}".format(instance.name), "wb") as f:
        #     pkl.dump(result, f)

    # print("Solving elevators instances")
    # for instance in elevators:
    #     upper = instance.countUncontrollables()
    #     result = epsilonConstraint(instance, "pstns/results/elevators/" upper, epsilon)
    #     with open("pstns/results/elevators/{}".format(instance.name), "wb") as f:
    #         pkl.dump(result, f)

    print("Finished")

if __name__ == "__main__":
    main()