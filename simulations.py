# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 23:06:10 2021

@author: kpb20194
"""
from tkinter import FALSE
import gurobipy as gp
from scipy.stats.stats import weightedtau
from scipy.stats import multivariate_normal
import pickle as pkl
from gurobipy import GRB
import os
import numpy as np
import sys
import convex
import monte_carlo as mc
import LinearProgram2 as LP
#import dill

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
    Description: Takes PSTN and gurobi solution and extracts schedule in form of dictionary of timepoint: value pairs.

    Input:  PSTN:       Instance to be solved
            solution:   Gurobi model containing solution

    Output: None:       None if error encountered
            schedule:   Dictionary {timepoint0: time,...,timepointn: value} of time-point: time pairs
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
    '''
    Description: Takes PSTN and gurobi solution and extracts schedule in form of dictionary of timepoint: value pairs.

    Input:  PSTN:           PSTN Instance to be solved
            solution:       Gurobi model containing solution

    Output: None:           None if error encountered
            relaxations:    dictionary {timepoint0: {Lower relaxation: value, Upper relaxation: value}..,timepointn: {Lower relaxation: value, Upper relaxation: value}}
    '''
    try:
        variables = [v.varName for v in solution.getVars()]
        values = [v.x for v in solution.getVars()]
        constraints = [c.name for c in PSTN.getConstraints() if c.type != "pstc"]
        relaxations = {}
        for c in constraints:
            relaxations[c] = {"Rl": 0, "Ru": 0}
        for i in range(len(variables)):
            if variables[i][-2:] == "rl":
                relaxations[variables[i][:-3]]["Rl"] = solution.getVarByName(variables[i]).x
            elif variables[i][-2:] == "ru":
                relaxations[variables[i][:-3]]["Ru"] = solution.getVarByName(variables[i]).x
        return relaxations
    except:
        return None

def main():
    # epsilon = 0.01
    # woodworking_path = "pstns/problems/woodworking"
    # woodworking_files = sorted(os.listdir(woodworking_path))
    # woodworking = []
    # for i in range(len(woodworking_files)):
    #    with open(woodworking_path + "/" + woodworking_files[i], "rb") as f:
    #        problem = pkl.load(f)
    #        woodworking.append(problem)

    # for i in range(len(woodworking_files)):
    #     print("\nSOLVING: ", woodworking[i].name, "\n")
    #     tosave = {}
    #     try:
    #         m, results = convex.solveJCCP2(woodworking[i], 0.2, 0.05, log=True, logfile=woodworking[i].name + "_woodworking_log")
    #         print("SOLVED: ", woodworking[i].name, "\n")
    #         schedule = getSchedule(woodworking[i], m)
    #         relaxations = getRelaxations(woodworking[i], m)
    #         tosave["PSTN"] = woodworking[i]
    #         tosave["JCCP"] = results
    #         tosave["Schedule"] = schedule
    #         tosave["Relaxations"] = relaxations
    #         #print(dill.detect.baditems(tosave))
    #         with open("results/{}_woodworking".format(woodworking[i].name), "wb") as f:
    #            pkl.dump(tosave, f)
    #     except:
    #        continue
        #try:
        #    m, results = LP.solveLP(woodworking[i], woodworking[i].name, 0.2)
        #    tosave = {}
        #    schedule = getSchedule(woodworking[i], m)
         #   relaxations = getRelaxations(woodworking[i], m)
          #  tosave["PSTN"] = woodworking[i]
           # tosave["LP"] = results
           # tosave["Schedule"] = schedule
           # tosave["Relaxations"] = relaxations
           # with open("results/{}_woodworking_LP".format(woodworking[i].name), "wb") as f:
            #    pkl.dump(tosave, f)
        #except:
         #   continue

        
    # elevators_path = "pstns/problems/elevators"
    # elevators_files = sorted(os.listdir(elevators_path))
    # elevators = []
    # for i in range(len(elevators_files)):
    #      with open(elevators_path + "/" + elevators_files[i], "rb") as f:
    #         problem = pkl.load(f)
    #         elevators.append(problem)


    # for i in range(13, 14):
    #     print("\nSOLVING: ", elevators[i].name, "\n")
    #     tosave = {}
    #     #try:
    #     m, results = convex.solveJCCP2(elevators[i], 0.2, 0.05, log=True, logfile=elevators[i].name + "_elevators_log")
    #     print("SOLVED: ", elevators[i].name, "\n")
    #     schedule = getSchedule(elevators[i], m)
    #     relaxations = getRelaxations(elevators[i], m)
    #     tosave["PSTN"] = elevators[i]
    #     tosave["JCCP"] = results
    #     tosave["Schedule"] = schedule
    #     tosave["Relaxations"] = relaxations
    #     #print(dill.detect.baditems(tosave))
    #     with open("results/{}_elevators".format(elevators[i].name), "wb") as f:
    #        pkl.dump(tosave, f)
        #except:
         #  continue
        # try:
        #     m, results = LP.solveLP(elevators[i], elevators[i].name, 0.2)
        #     tosave = {}
        #     schedule = getSchedule(elevators[i], m)
        #     relaxations = getRelaxations(elevators[i], m)
        #     tosave["PSTN"] = elevators[i]
        #     tosave["LP"] = results
        #     tosave["Schedule"] = schedule
        #     tosave["Relaxations"] = relaxations
        #     with open("results/{}_elevators_LP".format(elevators[i].name), "wb") as f:
        #         pkl.dump(tosave, f)
        # except:
        #     continue

    # print("Finished")


    cdru_path = "pstns/problems/cdru"
    cdru_files = sorted(os.listdir(cdru_path))
    cdru = []
    for i in range(len(cdru_files)):
         with open(cdru_path + "/" + cdru_files[i], "rb") as f:
            problem = pkl.load(f)
            cdru.append(problem)
            #problem.printPSTN()

    print([p.name for p in cdru])

    for i in range(len(cdru_files)):
        if i in [1]:
            cdru[i].plot()
    #         print("\nSOLVING: ", cdru[i].name, "\n")
    #         tosave = {}
    #         try:
    #             m, results = convex.solveJCCP2(cdru[i], 0.2, 0.05, log=True, logfile=cdru[i].name + "_log")
    #             print("SOLVED: ", cdru[i].name, "\n")
    #             schedule = getSchedule(cdru[i], m)
    #             relaxations = getRelaxations(cdru[i], m)
    #             tosave["PSTN"] = cdru[i]
    #             tosave["JCCP"] = results
    #             tosave["Schedule"] = schedule
    #             tosave["Relaxations"] = relaxations
    #         #print(dill.detect.baditems(tosave))
    #             with open("results/{}".format(cdru[i].name), "wb") as f:
    #                 pkl.dump(tosave, f)
    #         except:
    #             continue
    #         try:
    #             m, results = LP.solveLP(cdru[i], cdru[i].name, 0.2)
    #             tosave = {}
    #             schedule = getSchedule(cdru[i], m)
    #             relaxations = getRelaxations(cdru[i], m)
    #             tosave["PSTN"] = cdru[i]
    #             tosave["LP"] = results
    #             tosave["Schedule"] = schedule
    #             tosave["Relaxations"] = relaxations
    #             with open("results/{}_LP".format(cdru[i].name), "wb") as f:
    #                 pkl.dump(tosave, f)
    #         except:
    #             continue

    # print("Finished")

if __name__ == "__main__":
    main()