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
#import column_generation_norm
from gecco import gecco_algorithm
from column_generation_norm import solve
from gecco_class import gecco
import monte_carlo as mc
import LinearProgramParis as LP
#import dill

inf = 10000

gurobi_status = {1: "Loaded", 2: "Optimal", 3: "Infeasible", 4: "Inf or unb", 5: "Unbounded", 6: "Cutoff", 7: "Iteration Limit", 8: "Node Limit", 9: "Time Limit", 10: "Solution Limit", 11: "Interrupted", 12: "Numeric", 13: "Suboptimal", 14: "In Progress", 15: "User Obj Limit"}

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
    # This imports the PSTN files and loops through running them all. Risk bound can be changed through modifying risks.
    woodworking_path = "pstns/problems/woodworking"
    woodworking_files = sorted(os.listdir(woodworking_path))
    woodworking = []
    for i in range(len(woodworking_files)):
       with open(woodworking_path + "/" + woodworking_files[i], "rb") as f:
           problem = pkl.load(f)
           woodworking.append(problem)

    elevators_path = "pstns/problems/elevators"
    elevators_files = sorted(os.listdir(elevators_path))
    elevators = []
    for i in range(len(elevators_files)):
         with open(elevators_path + "/" + elevators_files[i], "rb") as f:
            problem = pkl.load(f)
            elevators.append(problem)

    for i in range(len(woodworking_files)):
        print("\nSOLVING: ", woodworking[i].name, "\n")
        try:
            tosave = {}
            m, results = solve(woodworking[i], 0.1, tolog=True, logfile = woodworking[i].name + "_woodworking_log_normal", max_iterations = 100, cg_tol = 0.05)
            print("SOLVED: ", woodworking[i].name, "\n")
            schedule = getSchedule(woodworking[i], m)
            tosave["PSTN"] = woodworking[i]
            tosave["Results"] = results
            tosave["Schedule"] = schedule
            with open("results/{}_woodworking_normal".format(woodworking[i].name), "wb") as f:
                pkl.dump(tosave, f)
        except:
            continue
        # try:
        #     tosave = {}
        #     m, results = gecco_algorithm(woodworking[i], tolog=True, logfile = woodworking[i].name + "_woodworking_log_genetic", max_iterations = 100)
        #     print("SOLVED: ", woodworking[i].name, "\n")
        #     schedule = getSchedule(woodworking[i], m)
        #     tosave["PSTN"] = woodworking[i]
        #     tosave["Results"] = results
        #     tosave["Schedule"] = schedule
        #     with open("results/{}_woodworking".format(woodworking[i].name), "wb") as f:
        #         pkl.dump(tosave, f)
        # except:
        #     continue
        # try:
        #     m, results = LP.solveLP(woodworking[i], woodworking[i].name)
        #     tosave = {}
        #     schedule = getSchedule(woodworking[i], m)
        #     tosave["PSTN"] = woodworking[i]
        #     tosave["LP"] = results
        #     tosave["Schedule"] = schedule
        #     with open("results/{}_woodworking_LP".format(woodworking[i].name), "wb") as f:
        #         pkl.dump(tosave, f)
        # except:
        #     continue
    # for i in range(len(elevators_files)):
    #     print("\nSOLVING: ", elevators[i].name, "\n")
    #     tosave = {}
    #     try:
    #         m, results = gecco_algorithm(elevators[i], tolog=True, logfile = elevators[i].name + "elevators_log_genetic", max_iterations = 100)
    #         print("SOLVED: ", elevators[i].name, "\n")
    #         schedule = getSchedule(elevators[i], m)
    #         tosave["PSTN"] = elevators[i]
    #         tosave["JCCP"] = results
    #         tosave["Schedule"] = schedule
    #         with open("results/{}_elevators".format(elevators[i].name), "wb") as f:
    #            pkl.dump(tosave, f)
    #     except:
    #         continue
    #     try:
    #         m, results = LP.solveLP(elevators[i], elevators[i].name)
    #         tosave = {}
    #         schedule = getSchedule(elevators[i], m)
    #         tosave["PSTN"] = elevators[i]
    #         tosave["LP"] = results
    #         tosave["Schedule"] = schedule
    #         with open("results/{}_elevators_LP".format(elevators[i].name), "wb") as f:
    #             pkl.dump(tosave, f)
    #     except:
    #         continue

    print("Finished")

if __name__ == "__main__":
    main()