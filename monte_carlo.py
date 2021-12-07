# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 08:43:53 2021

@author: kpb20194
"""
import numpy as np
    
def monte_carlo_success(PSTN, schedule, relaxation, no_simulations):
    '''
    Description: For a given schedule and PSTN, simulates execution of schedule a set amount of times and return probability of successfule execution (i.e. all constraints satisfied)
    Input:  PSTN - instance to be simulated
            schedule - schedule to be simulated
            no_simulations - number of times to simulate execution
    Output: no times successfully executed/total number of simulations
    '''
    if schedule == None:
        return None
    else:
        count = 0
        for i in range(no_simulations):
            if simulate_execution(PSTN, schedule, relaxation) == True:
                count += 1
        return count/no_simulations

def simulate_execution(PSTN, schedule, relaxation):
    '''
    Description: For a given schedule and PSTN, simulates execution of schedule once and returns True if successful (all constraints met) else returns False.
            PSTN - instance to be simulated
            schedule - schedule to be simulated
    Output: True (if all constraints met) else False
    '''
    contingents = PSTN.getContingents()
    for probabilistic in contingents:
        mean, sd = probabilistic.mu, probabilistic.sigma
        schedule[probabilistic.sink.id] = schedule[probabilistic.source.id] + np.random.normal(mean, sd)

    for constraint in PSTN.constraints:
        if constraint.type != "pstc":
            start, end = schedule[constraint.source.id], schedule[constraint.sink.id]
            try:
                if round(end - start, 10) < round(constraint.intervals["lb"] - relaxation[constraint.name]["Rl"], 10) or round(end - start, 10) > round(constraint.intervals["ub"] + relaxation[constraint.name]["Ru"], 10):
                    return False
            except KeyError:
                if round(end - start, 10) < round(constraint.intervals["lb"], 10) or round(end - start, 10) > round(constraint.intervals["ub"], 10):
                    return False
    return True