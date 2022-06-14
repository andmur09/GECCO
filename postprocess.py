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

    runtime_LP_elevators = []
    no_uncontrollables_LP_elevators = []
    cost_LP_elevators = []

    runtime_LP_wood = []
    no_uncontrollables_LP_wood = []
    cost_LP_wood = []

    runtime_elevators = []
    no_uncontrollables_elevators = []
    cost_elevators = []

    runtime_wood = []
    no_uncontrollables_wood = []
    cost_wood = []

    runtime_1it_elevators = []
    cost_1it_elevators = []

    runtime_1it_wood = []
    cost_1it_wood = []


    for i in files:
        if "elevators" in i:
            if "LP" in i:
                with open(results_path + "/" + i, "rb") as f:
                    instance = pkl.load(f)
                    try:
                        if "Objective" in instance["LP"].keys():
                            cost_LP_elevators.append(instance["LP"]["Objective"])
                            runtime_LP_elevators.append(instance["LP"]["Runtime"])
                            no_uncontrollables_LP_elevators.append(instance["PSTN"].countUncontrollables())
                    except:
                        continue
            else:
                with open(results_path + "/" + i, "rb") as f:
                    instance = pkl.load(f)
                    try:
                        cost_elevators.append(instance["JCCP"].solution["Objective"])
                        runtime_elevators.append(instance["JCCP"].solution_time)
                        no_uncontrollables_elevators.append(instance["PSTN"].countUncontrollables())
                        runtime_1it_elevators.append(instance["JCCP"].master_time[0][0])
                        cost_1it_elevators.append(instance["JCCP"].master_time[0][1])
                    except:
                        continue
        elif "woodworking" in i:
            if "LP" in i:
                with open(results_path + "/" + i, "rb") as f:
                    instance = pkl.load(f)
                    try:
                        if "Objective" in instance["LP"].keys():
                            cost_LP_wood.append(instance["LP"]["Objective"])
                            runtime_LP_wood.append(instance["LP"]["Runtime"])
                            no_uncontrollables_LP_wood.append(instance["PSTN"].countUncontrollables())
                    except:
                        continue
            else:
                with open(results_path + "/" + i, "rb") as f:
                    instance = pkl.load(f)
                    try:
                        cost_wood.append(instance["JCCP"].solution["Objective"])
                        runtime_wood.append(instance["JCCP"].solution_time)
                        no_uncontrollables_wood.append(instance["PSTN"].countUncontrollables())
                        runtime_1it_wood.append(instance["JCCP"].master_time[0][0])
                        cost_1it_wood.append(instance["JCCP"].master_time[0][1])
                    except:
                        continue

    # for i in files:
    #     if i == "p05_elevators_0.1":
    #         with open(results_path + "/" + i, "rb") as f:
    #             instance = pkl.load(f)
    #         times = [i[0] for i in instance["JCCP"].master_time]
    #         costs = [i[1] for i in instance["JCCP"].master_time]
    #     if i == "p05_elevators_LP_0.1":
    #         with open(results_path + "/" + i, "rb") as f:
    #             instance = pkl.load(f)
    #         LP_cost = instance["LP"]["Objective"]
    # cost_ratio = [i/LP_cost for i in costs]

    plt.rc('font', size=12)
    plt.rc('axes', titlesize=12)
    plt.rc('axes', labelsize=12)
    
    # plt.figure()
    # plt.plot(times, cost_ratio)
    # plt.xlabel("Time (s)")
    # plt.ylabel("JCC Cost/LP Cost")
    # plt.savefig("runtime_cost.png")

    print(sorted(cost_wood))
    print(sorted(cost_elevators))

    plt.figure()
    plt.scatter(no_uncontrollables_elevators, runtime_elevators, label="JCC elevators", marker ="x", color="blue")
    plt.scatter(no_uncontrollables_wood, runtime_wood, label="JCC woodworking", marker =".", color="blue")
    plt.scatter(no_uncontrollables_elevators, runtime_1it_elevators, label="JCC 1 elevators", marker = "x", color="green")
    plt.scatter(no_uncontrollables_wood, runtime_1it_wood, label="JCC 1 woodworking", marker = ".", color="green")
    plt.scatter(no_uncontrollables_LP_elevators, runtime_LP_elevators, label ="LP elevators", marker ="x", color="orange")
    plt.scatter(no_uncontrollables_LP_wood, runtime_LP_wood, label ="LP woodworking", marker =".", color="orange")
    plt.legend(fontsize=9)
    plt.xlabel("No of Uncontrollable Constraints")
    plt.yscale('log')
    plt.ylabel("Runtime (s)")
    plt.savefig("runtime3.png")

    plt.figure()
    plt.scatter(no_uncontrollables_elevators, cost_elevators, label="JCC elevators", marker="x", color="blue")
    plt.scatter(no_uncontrollables_wood, cost_wood, label="JCC woodworking", marker=".", color="blue")
    plt.scatter(no_uncontrollables_elevators, cost_1it_elevators, label = "JCC 1 elevators", marker = "x",color="green")
    plt.scatter(no_uncontrollables_wood, cost_1it_wood, label = "JCC 1 woodworking", marker = ".",color="green")
    plt.scatter(no_uncontrollables_LP_elevators, cost_LP_elevators, label="LP elevators", marker="x",color="orange")
    plt.scatter(no_uncontrollables_LP_wood, cost_LP_wood, label="LP woodworking", marker=".", color="orange")
    plt.legend(fontsize=9)
    plt.xlabel("No of Uncontrollable Constraints")
    plt.yscale('log')
    plt.ylabel("Cost")
    plt.savefig("cost3.png")

if __name__ == "__main__":
    main()