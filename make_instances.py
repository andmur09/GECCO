# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 08:39:09 2021

@author: kpb20194
"""
import sys
from ros_dot_parser import parseDot
from plan_duration import getPlanDuration
inf = 10000

def makeSTN(directory, no_instances):
    """ 
    This function iterates through a directory and parses dot files to make STN instances

    Input: directory - string of directory containint dot files

    Ouput: no_instances - integer of number of problem instances to be parsed
           names - list of file names "x" from "x".pddl.dot
    (Dot files must follow naming convention "p{}.pddl.dot")
    """
    files = []
    names = []
    deadlines = []
    for i in range(1, no_instances+1):
        if i < 10:
            fileName = "p{}.pddl.dot".format("0"+str(i))
            filePlan = "p{}.pddl_plan.txt".format("0"+str(i))
        else:
            fileName = "p{}.pddl.dot".format(str(i))
            filePlan = "p{}.pddl_plan.txt".format(str(i))
        names.append(fileName[:3])
        files.append(fileName)
        deadlines.append(getPlanDuration(directory+filePlan))
        
    problems = [directory+files[i] for i in range(len(files))]
    instances = [parseDot(problem) for problem in problems]
    return names, instances, deadlines

