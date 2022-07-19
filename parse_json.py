# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 16:11:06 2021

@author: kpb20194
"""
import json
from PSTN_class import PSTN, timePoint, constraint
import numpy as np
inf = np.inf

def JSONtoPSTN(filename, name):
# open the JSON file, parse and return PSTN instance
    with open(filename) as f:
        data = json.load(f)

    # Makes list of timepoints from nodes
    timepoints = [timePoint(t["node_id"], t["node_id"]) for t in data["nodes"]]
    # Adds start node
    timepoints.insert(0, timePoint(0, "Start"))

    constraints = []
    nodes = data["nodes"]
    for i in range(len(nodes)):
        if 'min_domain' in nodes[i]:
            intervals = {"lb": max(0, float(nodes[i]["min_domain"])), "ub": float(nodes[i]["max_domain"]),"value": 1}
            source, sink = timepoints[0], timepoints[i + 1]
            description = "c({},{})".format(str(source.id), str(sink.id))
            constraints.append(constraint(description, source, sink, "stc", intervals))
        else:
            intervals = {"lb": 0, "ub": inf, "value": 1}
            source, sink = timepoints[0], timepoints[i + 1]
            description = "c({},{})".format(str(source.id), str(sink.id))
            constraints.append(constraint(description, source, sink, "stc", intervals))
            
    problem = PSTN(name, timepoints, constraints)
    edges = data["constraints"]

    for c in edges:
        # Gets the source node associated with the constraint.
        source = [i for i in problem.timePoints if i.id == c["first_node"]]
        assert len(source) == 1
        source = source[0]
        sink = [j for j in problem.timePoints if j.id == c["second_node"]]
        assert len(sink) == 1
        sink = sink[0]
        existing_edge = problem.getConstraintByTimpoint(source.id, sink.id)
        if existing_edge != None:
            existing_edge.intervals = {"lb": max(0, float(c["min_duration"])), "ub": float(c["max_duration"]),"value": 1}
            if "distribution" in c:
                existing_edge.type = "pstc"
                dist = c["distribution"]["name"].split("_")
                existing_edge.distribution = {"type": "gaussian", "mean": float(dist[1]), "variance": float(dist[2])}
        else:
            if "distribution" in c:
                type = "pstc"
                description = "c({},{})".format(str(source.id), str(sink.id))
                intervals = {"lb": 0, "ub": inf, "value": 1}
                dist = c["distribution"]["name"].split("_")
                distribution = {"type": "gaussian", "mean": float(dist[1]), "variance": float(dist[2])}
            else:
                type = "stc"
                description = "c({},{})".format(str(source.id), str(sink.id))
                intervals = {"lb": max(0, c["min_duration"]), "ub": c["max_duration"],"value": 1}
                distribution = None
            problem.addConstraint(description, source, sink, type, intervals, distribution)

    for t in problem.timePoints:
        t.id = str(t.id)
        t.description = str(t.description)
    
    return problem

    # Adapted from Code:
    # def loadSTNfromJSONobj(jsonSTN, using_PSTN=True):
    # stn = STN()

    # # Add the root vertex and put it in the T_x set
    # stn.addVertex(0)

    # # Add the vertices
    # for v in jsonSTN['nodes']:
    #     stn.addVertex(v['node_id'])
    #     if 'min_domain' in v:       
    #         stn.addEdge(0, v['node_id'], float(v['min_domain']),
    #             float(v['max_domain']))
    #     else:
    #         if not stn.edgeExists(0, v['node_id']):
    #             stn.addEdge(0,v['node_id'], float(0), float('inf'))


    # # Add the edges
    # for e in jsonSTN['constraints']:
    #     if stn.edgeExists(e['first_node'], e['second_node']):
    #         stn.updateEdge(e['first_node'], e['second_node'],float(e['max_duration']))
    #         stn.updateEdge(e['second_node'], e['first_node'],float(e['min_duration']))
    #     else:
    #         if using_PSTN and 'distribution' in e:
    #             stn.addEdge(e['first_node'], e['second_node'],
    #                         float(max(0,e['min_duration'])), float(e['max_duration']),
    #                         e['distribution']['type'], e['distribution']['name'])
    #         elif 'type' in e:
    #             if e['type'] == 'stcu':
    #                 dist = "U_"+str(e['min_duration']) + "_" + str(e['max_duration'])
    #                 stn.addEdge(e['first_node'], e['second_node'],
    #                     float(max(0,e['min_duration'])), float(e['max_duration']),
    #                     e['type'], dist)
    #             else:
    #                 stn.addEdge(e['first_node'], e['second_node'],
    #                             float(e['min_duration']), float(e['max_duration']),
    #                             e['type'])
    #         else:
    #             stn.addEdge(e['first_node'], e['second_node'],
    #                         float(e['min_duration']), float(e['max_duration']))

    # return stn