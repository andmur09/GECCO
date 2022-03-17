# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:09:54 2021

@author: kpb20194
"""
import pickle as pkl
from matplotlib import pyplot as plt
import os
import monte_carlo as mc

def main():
    # Script used to generate plots of results
    
    results_path = "results"
    files = sorted(os.listdir(results_path))

    runtime_LP = []
    no_uncontrollables_LP = []
    cost_LP = []

    runtime = []
    no_uncontrollables = []
    cost = []

    runtime_1it = []
    cost_1it = []


    for i in files:
        if "elevators" in i or "woodworking" in i:
            if "LP" in i:
                with open(results_path + "/" + i, "rb") as f:
                    instance = pkl.load(f)
                    try:
                        if "Objective" in instance["LP"].keys():
                            cost_LP.append(instance["LP"]["Objective"])
                            runtime_LP.append(instance["LP"]["Runtime"])
                            no_uncontrollables_LP.append(instance["PSTN"].countUncontrollables())
                    except:
                        continue
            else:
                with open(results_path + "/" + i, "rb") as f:
                    instance = pkl.load(f)
                    try:
                        cost.append(instance["JCCP"].solution["Objective"])
                        runtime.append(instance["JCCP"].solution_time)
                        no_uncontrollables.append(instance["PSTN"].countUncontrollables())
                        runtime_1it.append(instance["JCCP"].master_time[0][0])
                        cost_1it.append(instance["JCCP"].master_time[0][1])
                    except:
                        continue

    for i in files:
        if i == "p05_elevators_0.1":
            with open(results_path + "/" + i, "rb") as f:
                instance = pkl.load(f)
            times = [i[0] for i in instance["JCCP"].master_time]
            costs = [i[1] for i in instance["JCCP"].master_time]
        if i == "p05_elevators_LP_0.1":
            with open(results_path + "/" + i, "rb") as f:
                instance = pkl.load(f)
            LP_cost = instance["LP"]["Objective"]
    cost_ratio = [i/LP_cost for i in costs]

    plt.rc('font', size=12)
    plt.rc('axes', titlesize=12)
    plt.rc('axes', labelsize=12)
    
    plt.figure()
    plt.plot(times, cost_ratio)
    plt.xlabel("Time (s)")
    plt.ylabel("JCC Cost/LP Cost")
    plt.savefig("runtime_cost.png")

    plt.figure()
    plt.scatter(no_uncontrollables, runtime, label="JCC", marker ="x")
    plt.scatter(no_uncontrollables, runtime_1it, label="JCC 1 Iteration", marker = ".")
    plt.scatter(no_uncontrollables_LP, runtime_LP, label ="LP", marker ="+")
    plt.legend()
    plt.xlabel("No of Uncontrollable Constraints")
    plt.yscale('log')
    plt.ylabel("Runtime (s)")
    plt.savefig("runtime2.png")

    plt.figure()
    plt.scatter(no_uncontrollables, cost, label="JCC", marker="x")
    plt.scatter(no_uncontrollables, cost_1it, label = "JCC 1 Iteration", marker = ".")
    plt.scatter(no_uncontrollables_LP, cost_LP, label="LP", marker="+")
    plt.legend()
    plt.xlabel("No of Uncontrollable Constraints")
    plt.yscale('log')
    plt.ylabel("Cost")
    plt.savefig("cost2.png")

if __name__ == "__main__":
    main()