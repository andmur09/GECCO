import PSTN_class as PSTN
from simulations import getSchedule, getRelaxations
import LinearProgram2 as LP
from tkinter import FALSE
import gurobipy as gp
from scipy.stats.stats import weightedtau
from scipy.stats import multivariate_normal
import pickle as pkl
from gurobipy import GRB
import os
import numpy as np
import column_generation_norm


timepoints = []
timepoints.append(PSTN.timePoint("0", "start_of_time"))
timepoints.append(PSTN.timePoint("1", "begin_plan_1"))
timepoints.append(PSTN.timePoint("2", "end_plan_1"))
timepoints.append(PSTN.timePoint("3", "begin_SLA"))
timepoints.append(PSTN.timePoint("4", "dummy"))
timepoints.append(PSTN.timePoint("5", "end_SLA"))
timepoints.append(PSTN.timePoint("6", "begin_plan_2"))
timepoints.append(PSTN.timePoint("7", "end_plan_2"))

constraints = []
constraints.append(PSTN.constraint("0 -> 1", timepoints[0], timepoints[1], "stc", {"lb": 0, "ub": 0, "value": -1}, hard=False))
constraints.append(PSTN.constraint("1 -> 2", timepoints[1], timepoints[2], "stc", {"lb": 10, "ub": 10, "value": 1}))
constraints.append(PSTN.constraint("2 -> 3", timepoints[2], timepoints[3], "stc", {"lb": 0, "ub": 1000, "value": 1}))
constraints.append(PSTN.constraint("0 -> 3", timepoints[0], timepoints[3], "pstc", {"lb": 0, "ub": 1000, "value": 1}, distribution={"type": "gaussian", "mean": 15, "variance": 1}))
constraints.append(PSTN.constraint("3 -> 4", timepoints[3], timepoints[4], "stc", {"lb": 0, "ub": 1000, "value": 1}))
constraints.append(PSTN.constraint("4 -> 5", timepoints[4], timepoints[5], "stc", {"lb": 0, "ub": 1000, "value": 1}))
constraints.append(PSTN.constraint("0 -> 5", timepoints[0], timepoints[4], "pstc", {"lb": 0, "ub": 1000, "value": 1}, distribution={"type": "gaussian", "mean": 18, "variance": 1}))
constraints.append(PSTN.constraint("5 -> 6", timepoints[4], timepoints[5], "stc", {"lb": 0, "ub": 0, "value": 1}, hard=False))
constraints.append(PSTN.constraint("6 -> 7", timepoints[5], timepoints[6], "stc", {"lb": 10, "ub": 10, "value": 1}))

problem = PSTN.PSTN("datacenter", timepoints, constraints)

risks = [0.6]
for risk in risks:
    tosave = {}
    try:
        m, results = column_generation_norm.solveJCCP(problem, risk, 0.1, log=True, logfile=problem.name + "_log")
        print("SOLVED: ", problem.name, "\n")
        schedule = getSchedule(problem, m)
        relaxations = getRelaxations(problem, m)
        tosave["PSTN"] = problem
        tosave["JCCP"] = results
        tosave["Schedule"] = schedule
        tosave["Relaxations"] = relaxations
        with open("results/{}_{}".format(problem.name, risk), "wb") as f:
            pkl.dump(tosave, f)
    except:
        continue
    try:
        m, results = LP.solveLP(problem, problem.name, risk)
        tosave = {}
        schedule = getSchedule(problem, m)
        relaxations = getRelaxations(problem, m)
        tosave["PSTN"] = problem
        tosave["LP"] = results
        tosave["Schedule"] = schedule
        tosave["Relaxations"] = relaxations
        with open("results/{}_LP_{}".format(problem.name, risk), "wb") as f:
            pkl.dump(tosave, f)
    except:
            continue