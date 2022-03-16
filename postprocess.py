# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:09:54 2021

@author: kpb20194
"""
from itertools import combinations_with_replacement
import pickle as pkl
from matplotlib import pyplot as plt
import os
import monte_carlo as mc

def plot_pareto(x, y, x_label, y_label, title):
    plt.figure()
    plt.plot(x, y, linestyle = "-", marker ="x", markersize = 6, linewidth=0.1)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.savefig("{}.png".format(title), bbox_inches='tight')

def main():
    ####################################################################
    ## ----------------------For woodworking cases--------------------##
    ####################################################################
    results_path = "results2"
    files = sorted(os.listdir(results_path))
    #print(woodworking_files)
    #woodworking_sol = [f for f in woodworking_files if f[-3:] not in ["sol", "ilp", "mps"]]

    runtime_LP = []
    no_uncontrollables_LP = []
    cost_LP = []

    no_uncontrollables_risk_LP = []
    risk_LP = []

    runtime = []
    no_uncontrollables = []
    cost = []

    no_uncontrollables_risk = []
    risk = []

    for i in files:
        #if "woodworking" in i:
        if "elevators" in i or "woodworking" in i:
            if "LP" in i:
                with open(results_path + "/" + i, "rb") as f:
                    instance = pkl.load(f)
                    try:
                        if "Objective" in instance["LP"].keys():
                            cost_LP.append(instance["LP"]["Objective"])
                            runtime_LP.append(instance["LP"]["Runtime"])
                            no_uncontrollables_LP.append(instance["PSTN"].countUncontrollables())
                        # if "0.2" in i:
                        #     no_uncontrollables_risk_LP.append(instance["PSTN"].countUncontrollables())
                        #     risk_LP.append(1 - mc.monte_carlo_success(instance["PSTN"], instance["Schedule"], instance["Relaxations"], 1000))
                    except:
                        continue
            else:
                with open(results_path + "/" + i, "rb") as f:
                    instance = pkl.load(f)
                    try:
                    #print(instance["JCCP"].master_time)
                        cost.append(instance["JCCP"].solution["Objective"])
                        runtime.append(instance["JCCP"].solution_time)
                        no_uncontrollables.append(instance["PSTN"].countUncontrollables())
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
    plt.scatter(no_uncontrollables, runtime, label="JCC")
    plt.scatter(no_uncontrollables_LP, runtime_LP, label ="LP")
    plt.legend()
    plt.xlabel("No of Uncontrollable Constraints")
    plt.yscale('log')
    plt.ylabel("Runtime (s)")
    plt.savefig("runtime2.png")

    plt.figure()
    plt.scatter(no_uncontrollables, cost, label="JCC")
    plt.scatter(no_uncontrollables_LP, cost_LP, label="LP")
    plt.legend()
    plt.xlabel("No of Uncontrollable Constraints")
    plt.yscale('log')
    plt.ylabel("Cost")
    plt.savefig("cost2.png")


    # delta = [(cost_LP[i] - cost[i])/cost[i] for i in range(len(cost_LP))]
    # print(cost)
    # print(cost_LP)
    # print(delta)
    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    # ax1.scatter(no_uncontrollables, cost, label="JCC")
    # ax1.scatter(no_uncontrollables_LP, cost_LP, label="LP")
    # ax2.scatter(no_uncontrollables, delta, label="LP-JCC")
    # ax1.set_yscale('log')
    # ax1.set_xlabel('No of Uncontrollable Constraints')
    # ax1.set_ylabel('Relaxation Cost')
    # ax2.set_ylabel('Cost Delta ((LP - JCC)/JCC)')
    # plt.legend()
    # plt.savefig("cost2.png", bbox_inches='tight')


    # plt.figure()
    # plt.scatter(no_uncontrollables, cost, label="JCC")
    # plt.scatter(no_uncontrollables_LP, cost_LP, label="LP")
    # plt.scatter(no_uncontrollables, delta, label="LP-JCC")
    # plt.legend()
    # plt.xlabel("No of Uncontrollable Constraints")
    # plt.yscale('log')
    # plt.ylabel("Relaxation Cost")
    # plt.savefig("cost2.png")

    # plt.figure()
    # plt.scatter(no_uncontrollables_risk, risk, label="JCCP")
    # plt.scatter(no_uncontrollables_risk_LP, risk_LP, label="LP")
    # plt.legend()
    # plt.xlabel("No of uncontrollables")
    # plt.ylabel("Risk")
    # plt.savefig("risk.png")

         #   pstn, result = instance["PSTN"], instance["Result"]
         #   for k in result:
         #       result[k]["Probability"] = mc.monte_carlo_success(pstn, result[k]["Schedule"], result[k]["Relaxations"], 1000)
         #   risk = [1 - result[k]["Probability"] for k in result]
          #  booles = [result[k]["Risk"] for k in result]
          #  print(risk)
          #  cost = [result[k]["Cost"] for k in result]
          #  plot_pareto(booles, cost, "Booles Risk", "Cost", "Booles_vs_Cost")
          #  plot_pareto(risk, cost, "Monte-Carlo Risk", "Cost", "Risk_vs_Cost")


    







    # woodworking_solved = []
    # for solution in woodworking_sol:
    #     result_path = solution[:-4] + "_result"
    #     result = getProbability(solution, result_path, woodworking_path)
    #     result = getRelaxationBudget(solution, result)
    #     result = getWeight(solution, result)
    #     result["No Constraints"] = len(result["PSTN"].constraints)
    #     woodworking_solved.append(result)

    # woodworking_per_problem = {}
    # woodworking_keys = ["p{:02d}".format(i) for i in range(1,31)]

    # for i in range(len(woodworking_keys)):
    #     woodworking_per_problem[woodworking_keys[i]] = []
    
    # for j in woodworking_solved:
    #     for k in woodworking_per_problem.keys():
    #         if j["PSTN"].name[:3] == k:
    #             woodworking_per_problem[k].append(j)

    # for k in woodworking_per_problem:
    #     problems = woodworking_per_problem[k]
    #     names = [i["PSTN"].name for i in problems if i["Weight"] == 0]     
    #     risks = [round(1- i["LP Probability"], 3)for i in problems if i["Weight"] == 0]
    #     dictionary = dict(zip(names, risks))
    #     print("\n", k, dictionary)

    # plt.figure()
    # for k in woodworking_per_problem:
    #     problems = woodworking_per_problem[k]
    #     risks = [1 - i["LP Probability"] for i in problems if i["Weight"] == 0.0]
    #     budgets = [i["Budget"] for i in problems if i["Weight"] == 0.0]
    #     plt.plot(budgets, risks, linestyle = "-", marker ="x", markersize = 6, linewidth=0.1)
    # plt.xlabel("Relaxation Budget Fraction")
    # plt.ylabel("Probability of Failure")
    # plt.savefig("risk_vs_relaxation_woodworking.png", bbox_inches='tight')

    # ####################################################################
    # ## -----------------------For elevators cases---------------------##
    # ####################################################################
    # elevators_path = "pstns_relaxation/results/elevators_2"
    # elevators_files = sorted(os.listdir(elevators_path))
    # elevators_sol = [f for f in elevators_files if f[-3:] == "sol"]

    # elevators_solved = []
    # for solution in elevators_sol:
    #     result_path = solution[:-4] + "_result"
    #     result = getProbability(solution, result_path, elevators_path)
    #     result = getRelaxationBudget(solution, result)
    #     result = getWeight(solution, result)
    #     result["No Constraints"] = len(result["PSTN"].constraints)
    #     elevators_solved.append(result)
    # elevators_per_problem = {}
    # elevators_keys = ["p{:02d}".format(i) for i in range(1,31)]

    # for i in range(len(elevators_keys)):
    #     elevators_per_problem[elevators_keys[i]] = []
    
    # for j in elevators_solved:
    #     for k in elevators_per_problem.keys():
    #         if j["PSTN"].name[:3] == k:
    #             elevators_per_problem[k].append(j)

    # plt.figure()
    # for k in elevators_per_problem.keys():
    #     problems = elevators_per_problem[k]
    #     try:
    #         risks = [1 - i["LP Probability"] for i in problems if i["Weight"] == 0]
    #         budgets = [i["Budget"] for i in problems if i["Weight"] == 0]
    #         plt.plot(budgets, risks, linestyle = "-", marker ="x", markersize = 6, linewidth=0.1)
    #     except KeyError:
    #         pass
    # plt.xlabel("Relaxation Budget Fraction")
    # plt.ylabel("Probability of Failure")
    # plt.xlim(0.6, 2)
    # plt.savefig("risk_vs_relaxation_elevators.png", bbox_inches='tight')

    # fig, ax = plt.subplots()
    # elevators_runtimes = [i["LP Runtime"] for i in elevators_solved if i["Weight"] == 0]
    # elevators_no_probabilistic = [i["No Constraints"] for i in elevators_solved if i["Weight"] == 0]
    # woodworking_runtimes = [i["LP Runtime"] for i in woodworking_solved if i["Weight"] == 0]
    # woodworking_no_probabilistic = [i["No Constraints"] for i in woodworking_solved if i["Weight"] == 0]
    # ax.scatter(elevators_no_probabilistic, elevators_runtimes, label = "Elevators", marker ="x", s = 10, linewidths=0.5)
    # ax.scatter(woodworking_no_probabilistic, woodworking_runtimes, label = "Woodworking", marker ="+", s = 10, linewidths=0.5)
    # plt.xlabel("Number of Constraints")
    # plt.ylabel("Runtime (s)")
    # plt.legend()
    # plt.savefig("runtime_both.png", bbox_inches='tight')

    # ## For all ##
    # fig, ax = plt.subplots()
    # elevators_runtimes = [i["LP Runtime"] for i in elevators_solved if i["Weight"] == 0]
    # elevators_relaxation = [i["Budget"] for i in elevators_solved if i["Weight"] == 0]
    # woodworking_runtimes = [i["LP Runtime"] for i in woodworking_solved if i["Weight"] == 0]
    # woodworking_relaxation = [i["Budget"] for i in woodworking_solved if i["Weight"] == 0]
    # ax.scatter(elevators_relaxation, elevators_runtimes, label = "Elevators", marker ="x", s = 10, linewidths=0.5)
    # ax.scatter(woodworking_relaxation, woodworking_runtimes, label = "Woodworking", marker ="+", s = 10, linewidths=0.5)
    # plt.xlabel("Relaxation Budget Fraction")
    # plt.ylabel("Runtime (s)")
    # plt.legend()
    # plt.savefig("runtime_relaxation.png", bbox_inches='tight')

if __name__ == "__main__":
    main()