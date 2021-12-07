# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 16:11:06 2021

@author: kpb20194
"""
import re
import sys
import PSTN_class as PSTN
inf = 10000

def parseDot(filename):
    """ 
    This function parses dot files and outputs an STN.

    Input: filename - string representing filename of dot file to be parsed.

    Ouput: List of dictionaries representing an STN. Each item in the list is a dictionary representing a constraint between
    two time points.
    """
    data1 = []
    data2 = []
    with open(filename, "r", errors="ignore") as dot:
        for line in dot:
            if re.search("\d+\[\slabel=", line) or ";" in line:
                data1.append(line.strip())
            elif "->" in line:
                data2.append(line.strip())

    nodes = []
    for i in range(len(data1)):
        if re.search("\d+\[\slabel=", data1[i]) and ";" in data1[i]:
            nodes.append(data1[i])
        elif re.search("\d+\[\slabel=", data1[i]):
            toAdd = data1[i]
            count = 1
            while ";" not in data1[i+count]:
                toAdd += data1[i+count]
                count += 1
            toAdd += data1[i+count]
            nodes.append(toAdd)
        else:
            pass

    timepoints = []
    for nodes in nodes:
        name = re.search("\d+\[", nodes).group()[:-1]
        description = re.search('label=\".+\"', nodes).group()[7:-1]
        timepoints.append(PSTN.timePoint(name, description))

    
    constraints = []
    for edge in data2:
        toAdd = {}
        split = edge.split()
        source_id, sink_id = split[0].strip('"'), split[2].strip('"')
        toAdd["description"] = source_id + " -> " + sink_id

        for timepoint in timepoints:
            if timepoint.id == source_id:
                toAdd["source"] = timepoint
            elif timepoint.id == sink_id:
                toAdd["sink"] = timepoint
        
        bounds = re.search('label=\"\[.+\]\"', edge).group()[8:-2].split(", ")
        lb, ub = bounds[0], bounds[1]
        if ub == "inf":
            print("Adding deadline")
            ub = inf
        toAdd["lb"], toAdd["ub"] = float(lb), float(ub)
        constraints.append(PSTN.constraint(toAdd["description"], toAdd["source"], toAdd["sink"], "stc", {"lb": toAdd["lb"], "ub": toAdd["ub"], "value": 1}))
    
    return (timepoints, constraints)