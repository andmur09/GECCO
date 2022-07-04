# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 09:20:02 2021

@author: kpb20194
"""
from make_instances import makeSTN
import sys
from PSTN_class import PSTN
import pickle as pkl
import numpy as np
import logging
import re
inf = 10000

def main():
    # logging.basicConfig(filename='elevators_generation.log', level=logging.INFO)
    directory = "pstns/domains/tfd-benchmarks/elevators-numeric/"
    no_instances = 30
    names, instances, deadlines = makeSTN(directory, no_instances)

    if len(names) != len(instances) or len(names) != len(deadlines):
        raise ValueError("Number of instances/names/deadlines not equal")
    
    # Makes instances of PSTN class using set of timepoints and set of constraints.
    elevators = []
    for i in range(no_instances):
        name = names[i]
        timepoints = instances[i][0]
        constraints = instances[i][1]
        elevators.append(PSTN(name, timepoints, constraints))
    
    for i in range(len(deadlines)):
        if deadlines[i] != None:
            deadlines[i] = deadlines[i]

    # For each PSTN, finds the start and last timePoint in the network and thus the constraint bounding the overall plan duration, creates additional instances of
    # each PSTN with varying deadlines.
    factors = [1.0, 1.2, 1.4, 1.6, 1.8, 1.2]
    elevators_ud = []
    for i in range(len(elevators)):
        for j in factors:
            numbers = str(j).split(".")
            problem = elevators[i].makeCopy(elevators[i].name + "_" + numbers[0] + numbers[1])
            deadline = deadlines[i]
            if deadline != None:
                problem.addDeadline(deadline*j)
                elevators_ud.append(problem)

    # Changes move actions to be probabilistic in PSTN
    for instance in elevators_ud:
        constraints = instance.constraints
        for constraint in constraints:
            if "move-up-slow_start" in constraint.source.description and "move-up-slow_end" in constraint.sink.description:
                if re.search("\(.+\)", constraint.source.description).group() == re.search("\(.+\)", constraint.sink.description).group():
                    constraint.setType("pstc")
                    mean = (constraint.intervals["lb"] + constraint.intervals["ub"])/2
                    constraint.distribution = {"type": "gaussian", "mean": mean, "variance": 0.2 * mean}
                    constraint.intervals = {"lb": 0, "ub": inf, "value": 1}
            elif "move-up-fast_start" in constraint.source.description and "move-up-fast_end" in constraint.sink.description:
                if re.search("\(.+\)", constraint.source.description).group() == re.search("\(.+\)", constraint.sink.description).group():
                    constraint.setType("pstc")
                    mean = (constraint.intervals["lb"] + constraint.intervals["ub"])/2
                    constraint.distribution = {"type": "gaussian", "mean": mean, "variance": 0.2 * mean}
                    constraint.intervals = {"lb": 0, "ub": inf, "value": 1}
            elif "move-down-slow_start" in constraint.source.description and "move-down-slow_end" in constraint.sink.description:
                if re.search("\(.+\)", constraint.source.description).group() == re.search("\(.+\)", constraint.sink.description).group():
                    constraint.setType("pstc")
                    mean = (constraint.intervals["lb"] + constraint.intervals["ub"])/2
                    constraint.distribution = {"type": "gaussian", "mean": mean, "variance": 0.2 * mean}
                    constraint.intervals = {"lb": 0, "ub": inf, "value": 1}
            elif "move-down-fast_start" in constraint.source.description and "move-down-fast_end" in constraint.sink.description:
                if re.search("\(.+\)", constraint.source.description).group() == re.search("\(.+\)", constraint.sink.description).group():
                    constraint.setType("pstc")
                    mean = (constraint.intervals["lb"] + constraint.intervals["ub"])/2
                    constraint.distribution = {"type": "gaussian", "mean": mean, "variance": 0.2 * mean}
                    constraint.intervals = {"lb": 0, "ub": inf, "value": 1}

    for i in range(len(elevators_ud)):
        with open("pstns/problems/elevators/{}".format(elevators_ud[i].name), "wb") as f:
            pkl.dump(elevators_ud[i], f)

if __name__ == '__main__':
    main()
