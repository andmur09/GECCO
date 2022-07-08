# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 08:43:53 2021

@author: kpb20194
"""
import numpy as np
    
def monte_carlo_success(PSTN, schedule, no_simulations):
    '''
    Description:    For a given schedule and PSTN, simulates execution of schedule a set amount of times and return probability
                    of successfule execution (i.e. all constraints satisfied)
    
    Input:      PSTN:           instance of PSTN class to be simulated
                schedule:       dictionary {timepoint0: time,...,timepointn: value} of time-point: time pairs
                relaxation:     dictionary {timepoint0: {Lower relaxation: value, Upper relaxation: value}..,timepointn: {Lower relaxation: value, Upper relaxation: value}}
                no_simulations: number of times to simulate execution
    
    Output:     float:          no times successfully executed/total number of simulations
    '''
    if schedule == None:
        return None
    else:
        count = 0
        for i in range(no_simulations):
            if simulate_execution(PSTN, schedule) == True:
                count += 1
        return count/no_simulations

def simulate_execution(PSTN, schedule):
    '''
    Description: For a given schedule and PSTN, simulates execution of schedule once and returns True if successful (all constraints met)
                 else returns False.
    
    Input:      PSTN:           instance of PSTN class to be simulated
                schedule:       dictionary {timepoint0: time,...,timepointn: value} of time-point: time pairs
                relaxation:     dictionary {timepoint0: {Lower relaxation: value, Upper relaxation: value}..,timepointn: {Lower relaxation: value, Upper relaxation: value}}
    
    Output:     bool:           True if successfully executed else False
    '''
    mean = PSTN.getMean()
    cov = PSTN.getCovariance()
    pts = np.random.multivariate_normal(mean, cov, size=1)[0]
    contingents = PSTN.getContingents()
    print([i.description for i in contingents])
    print(mean)
    print(cov)
    print(pts)

    for i in range(len(contingents)):
        outcome = pts[i]
        schedule[contingents[i].sink.id] = schedule[contingents[i].source.id] + outcome
    print(schedule)
    for constraint in PSTN.constraints:
        if constraint.type != "pstc":
            start, end = schedule[constraint.source.id], schedule[constraint.sink.id]
            print("\nEnd = {} {}, start = {} {}".format(constraint.sink.id, end, constraint.source.id, start))
            print("Value = {}, LB = {}, UB = {}".format(round(end - start, 10), round(constraint.intervals["lb"], 10), round(constraint.intervals["ub"], 10)))
            if round(end - start, 10) < round(constraint.intervals["lb"], 10) or round(end - start, 10) > round(constraint.intervals["ub"], 10):
                #print("Condition not met")
                return False
    return True