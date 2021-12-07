# -*- coding: utf-8 -*-
"""
Created on Wed Feb  3 11:55:20 2021

@author: kpb20194
"""
import sys
from graphviz import Digraph
import subprocess
import copy
import logging
inf = 10000

def key_exists(dictionary, keys):
    ## Check if *keys (nested) exists in `element` (dict).
    if not isinstance(dictionary, dict):
        raise AttributeError('keys_exists() expects dict as first argument.')
    if len(keys) == 0:
        raise AttributeError('keys_exists() expects at least two arguments, one given.')

    _dictionary = dictionary
    for key in keys:
        try:
            _dictionary = _dictionary[key]
        except KeyError:
            return False
    return True 

class timePoint(object):
    # newName = itertools.count()
    ## Class representing a PSTN time-point
    def __init__(self, name, description, controllable = None):
        self.id = name
        self.description = description
        self.controllable = controllable
        # self.id = next(timePoint.newName)
        # self.name = "t({})".format(str(self.id))
        
    def setControllable(self, logic):
    ## Used to set attribute for timepoint. If logic == True, timepoint is controllable, if logic == False, timepoint is uncontrollable
        self.controllable = logic
    
    def isControllable(self):
        if self.controllable == True:
            return True
        elif self.controllable == False:
            return False
        else:
            raise AttributeError("Controllable attribute not set for time-point")
    
    def getName(self):
        return self.name
    
    def __str__(self):
        return "Time-point {}".format(self.id)
    
    def copy(self):
        return timePoint(self.description[:])
    
class constraint(object):
    ## Class representing a constraint
        # \param description    String name describing constraint
        # \param soure          Start node in the constraint (instance of timePoint class)
        # \param sink           End node in the constraint (instance of timePoint class)
        # \param intervals      Possible duration intervals for constraint, this should be a list of dictionaries in the form [{"lb": interval1_lb, "ub": interval1_ub, "value": interval1_value},...]
        # \param type           Type of constraint: stc (requirement constraint), stcu (set-bounded contingent link), pstc (probabilistic constraint)
        # \param distribution   Dictionary of distribution properties: None if not probabilistic, {type: uniform or gaussian, mean: mu, variance: sigma}
        # \param cost           Represents allowable relaxation of constraint. Defaults to 0 meaning constraints are hard constraints
    def __init__(self, description, source, sink, type, intervals, distribution = None, hard = True):
        self.description = description
        self.source = source
        self.sink = sink
        self.intervals = intervals
        self.distribution = distribution
        self.hard = hard
        if type not in  ("stc", "stcu", "pstc"):
            raise TypeError("Invalid Constraint type, type must be 'stc' for simple temporal constraint, 'stcu' for simple temporal constraint with uncertainty or 'pstc' for probabilistic simple temporal constraint")
        else:
            self.type = type
        self.name = "c({},{})".format(str(source.id), str(sink.id))

    def copyConstraint(self):
        return constraint(self.description[:], self.source, self.sink, self.type[:], copy.deepcopy(self.intervals), distribution = copy.deepcopy(self.distribution), hard = copy.deepcopy(self.hard))

    def setType(self, _type):
        if _type not in  ("stc", "stcu", "pstc"):
            raise TypeError("Invalid Constraint type")
        else:
            self.type = _type

    def getType(self):
        return self.type
        
    def setDistribution(self, distribution):
        self.distribution = distribution

    def getDistribution(self):
        return self.distribution

    def dtype(self):
        if self.distribution["type"] == "uniform":
            return "Uniform"
        if self.distribution["type"] == "gaussian":
            return "Gaussian"
        else:
            return "Distribution type not currently supported, distribution['type'] must be 'uniform' or 'gaussian'"
    
    def __str__(self):
        if self.type == "stc" or self.type == "stcu":
            result = "Edge {} --> {}: ".format(self.source, self.sink)
            for i in self.intervals:
                result.append("[{}, {}] ".format(i["lb"], i["ub"]))
        elif self.type == "pstc":
            result = "Edge {} --> {}: N({}, {})".format(self.source, self.sink, self.distribution["mean"], self.distribution["variance"])
        return result

    def forJSON(self):
        jsonDict = {"end_event_name": self.sink, "type": self.type, "name": self.description}
        jsonDict["properties"] = {}
        if self.type == "stc" or self.type == "stcu":
            jsonDict["properties"]["lb"] = self.lb
            jsonDict["properties"]["ub"] = self.ub
        elif self.type == "pstc":
            if self.distribution["type"] == "uniform":
                jsonDict["properties"]["distribution"] = {"type": "Uniform", "lb": self.dist_lb, "ub": self.dist_ub}
            elif self.distribution["type"] == "gaussian":
                jsonDict["properties"]["distribution"] = {"type": "Gaussian", "mean": self.mu, "variance": self.sigma}
        jsonDict["start_event_name"] = self.source
        return jsonDict

    @property
    def mu(self):
        if self.distribution != None and self.distribution["type"] == "gaussian":
            return self.distribution["mean"]
        else:
            raise ValueError

    @property
    def sigma(self):
        if self.distribution != None and self.distribution["type"] == "gaussian":    
            return self.distribution["variance"]
        else:
            raise ValueError
    
    @property
    def dist_ub(self):
        if self.distribution != None and self.distribution["type"] == "uniform":
            return self.distribution["ub"]
        else:
            raise ValueError
        
    @property
    def dist_lb(self):
        if self.distribution != None and self.distribution["type"] == "uniform":
            return self.distribution["lb"]
        else:
            raise ValueError
    

class PSTN(object):
    def __init__(self, name, timePoints, constraints, adjList = None):
        ## Class representing a PSTN
        # \param name               String name of PSTN
        # \param timePoints         List of instances of timePoint class
        # \param constraints        List of instances of constraint class
        self.name = name
        self.timePoints = timePoints
        self.constraints = constraints
        self.adjList = adjList

    def setName(self, name):
        self.name = name
    
    def makeCopy(self, name):
        return PSTN(name,  self.timePoints[:], [constraint.copyConstraint() for constraint in self.constraints])

    def isSTN(self):
        ## Checks to see if the instance is an STN, i.e. it only contains simple temporal constraints.
        ## Returns True if STN, False otherwise
        for constraint in self.constraints:
            if constraint.getType() != "stc" or len(constraint.intervals) > 1:
                print("Not an STN")
                return False
        return True

    def getAdjacencyList(self):
        ## Returns an adjacency list giving the temporal distance between each node in the STN. Sets self distances to 0 and initialises missing distances as infinity
        adjList = {}
        for source in self.timePoints:
            adjList[source.id] = {}
            for sink in self.timePoints:
                constraint = self.getConstraintByTimpoint(source.id, sink.id)
                if constraint == None:
                    if source.id == sink.id:
                        adjList[source.id][sink.id] = 0
                    else:
                        adjList[source.id][sink.id] = inf
                else:
                    if constraint.sink.id == sink.id:
                        adjList[source.id][sink.id] = constraint.intervals["ub"]
                    elif constraint.source.id == sink.id:
                        adjList[source.id][sink.id] = -constraint.intervals["lb"]
        self.adjList = adjList

    def makeMinimal(self):
        minimal = self.makeCopy(self.name[:])
        timePoints = [i.id for i in self.timePoints]
        for k in timePoints:
            for i in timePoints:
                for j in timePoints:
                    if self.adjList[i][j] > self.adjList[i][k] + self.adjList[k][j]:
                        self.adjList[i][j] = self.adjList[i][k] + self.adjList[k][j]
                        constraint = minimal.getConstraintByTimpoint(i, j)
                        if constraint != None and constraint.getType() != "pstc":
                            if constraint.source.id == i:
                                constraint.intervals["ub"] = self.adjList[i][k] + self.adjList[k][j]
                            elif constraint.sink.id == i:
                                constraint.intervals["lb"] = -1* (self.adjList[i][k] + self.adjList[k][j])

        for constraint in minimal.getConstraints():
            if constraint.intervals["lb"] > constraint.intervals["ub"]:
                return False

        return minimal

    def checkConsistency(self):
        ## Floyd-Warshall algorithm to check consistency of STN
        ## Updates STN to all pairs shortest path and returns True if consistent, else if detects negative cycles then not consistent and returns False
        timepoints = [i.id for i in self.timePoints]
        for k in timepoints:
            for i in timepoints:
                for j in timepoints:
                    if self.adjList[i][j] > self.adjList[i][k] + self.adjList[k][j]:
                        # print("Distance from Node {} to {} updated from {} to {}, since distance {}->{} > {}->{} + {}->{}".format(i, j, self.adjList[i][j], self.adjList[i][k] + self.adjList[k][j], i, j, i, k, k, j))
                        self.adjList[i][j] = self.adjList[i][k] + self.adjList[k][j]
                    if i==j and self.adjList[i][j] < 0:
                        return False
        return True

    
    def formatSTN(self, plot = False):
        # Makes an adjacency list of STN
        self.getAdjacencyList()
        if plot == True:
            self.plotAdjList(self.name + "_initial")
        # Performs Floyd-Warshall to check consistency and return all-pairs shortest path
        minimal = self.makeMinimal()
        # If negative cycles are detected, STN is not consistent
        if minimal == False:
            print("PSTN not consistent")
            return False
        else:
            print("All pairs shortest path successful")
            if plot == True:
                self.plotAdjList(self.name + "_apsp")
            return minimal

    def isControllable(self, constraint):
        ## Checks if the requirement constraint is controllable (i.e. contains no uncontrollable time-points)
        ## Returns True if controllable, False otherwise
        tc = self.getControllables()
        tu = self.getUncontrollables()
        if constraint.source.isControllable() == True and constraint.sink.isControllable() == True:
            return True
        else:
            return False
    
    def incomingContingent(self, constraint):
        if self.isControllable(constraint) == True:
            raise AttributeError("Constraint has no incoming contingent links")
        if self.isControllable(constraint) == False:
            incoming_source = [g for g in self.getContingents() if g.sink == constraint.source]
            incoming_sink = [g for g in self.getContingents() if g.sink == constraint.sink]
            if len(incoming_source) > 1 or len(incoming_sink) > 1:
                raise AttributeError("More than one incoming contingent edge")
            else:
                try:
                    return {"start": incoming_source[0], "end": incoming_sink[0]}
                except IndexError:
                    try:
                        return {"start": incoming_source[0], "end": None}
                    except IndexError:
                        return {"start": None, "end": incoming_sink[0]}
    
    def outgoingEdge(self, constraint):
        return [c for c in self.getConstraints() if c.source == constraint.sink]

    def getConstraints(self):
        return self.constraints
    
    def getConstraintByTimpoint(self, source_id, sink_id):
        result = None
        for constraint in self.constraints:
            if constraint.source.id == source_id and constraint.sink.id == sink_id:
                result = constraint
            elif constraint.sink.id == source_id and constraint.source.id == sink_id:
                result = constraint
        return result

    def addConstraint(self, description, source, sink, type, intervals, distribution = None, hard = True):
        new_constraint = constraint(description, source, sink, type, intervals, distribution = distribution, hard = hard)
        self.constraints.append(new_constraint)
        return new_constraint

    def addTimePoint(self, timePoint):
        self.timePoints.append(timePoint)
    
    def addDeadline(self, deadline):
        final_tps = set()
        for constraint in self.getConstraints():
            if len(self.outgoingEdge(constraint)) < 1:
                final_tps.add(constraint.sink)
        start_tp_id = str(min([int(timepoint.id) for timepoint in self.getTimePoints()]))
        end_tp_id = str((max([int(timepoint.id) for timepoint in self.getTimePoints()])) + 1)
        eot = timePoint(end_tp_id, "Dummy timepoint to enforce deadline")
        self.addTimePoint(eot)
        self.addConstraint(start_tp_id + " -> " + end_tp_id, self.getTimePointByID(start_tp_id), eot, "stc", {"lb": 0, "ub": deadline, "value": 1}, hard = False)
        for tp in final_tps:
            self.addConstraint(tp.id + " -> " + end_tp_id, tp, eot, "stc", {"lb": 0.001, "ub": inf, "value": 1})
        print("Deadline {} added".format(deadline))

    def getTimePoints(self):
        return self.timePoints
    
    def getTimePointByID(self, id):
        for timepoint in self.getTimePoints():
            if id == timepoint.id:
                return timepoint
        print("No timepoint found with that ID.")
        return False

    def getRequirements(self):
        return [i for i in self.constraints if i.type == "stc"]
        
    def getContingents(self):
        return [i for i in self.constraints if i.type != "stc"]
        
    def getUncontrollables(self):
        tu = [i.sink for i in self.getContingents()]
        for t in tu:
            t.setControllable(False)
        return tu

    def getUncontrollableConstraints(self):
        return [i for i in self.constraints if self.isControllable(i) == False and i.type != "pstc"]

    def getControllables(self):
        tc = [i for i in self.timePoints if i not in self.getUncontrollables()]
        for t in tc:
            t.setControllable(True)
        return tc

    def getControllableConstraints(self):
        return [i for i in self.constraints if self.isControllable(i) == True]

    def setConstraints(self, constraints):
        self.constraints = constraints
    
    def setTimePoints(self, timepoints):
        self.timepoints = timepoints
        
    def getSize(self):
        return len(self.getConstraints())

    def pstnJSON(self):
        constList = []
        for constraint in self.constraints:
            constList.append(constraint.forJSON())
        toWrite = {self.name: constList}
        try:
            f = open("{}.json".format(self.name), "x")
            f.write(str(toWrite))
            f.close()
            print("File successfully created")
            return True
        except FileExistsError:
            print("File name already in use. Try changing name of PSTN to something not in use")
            answer = input("Change name of PSTN? [Y/N]: ")
            if answer == "Y" or answer == "y":
                newName = input("Input new name: ")
                self.setName(newName)
                self.pstnJSON()
            else:
                print("File creation failed")
                return False
            
    def plot(self, dp = 3):
        """
        Input
        -------
        A PSTN instance
        
        Returns
        -------
        Prints the PSTN in dot format and produces a PDF of the graph.

        """
        plot = Digraph()
        requirements = self.getRequirements()
        contingents = self.getContingents()

        for timePoint in self.timePoints:
            plot.node(name=timePoint.id, label=str(timePoint.id))
        
        for constraint in requirements:
            intervals = "[" + str(round(constraint.intervals["lb"], dp)) + ", " + str(round(constraint.intervals["ub"], dp)) + "] "
            if constraint.hard == True:
                relaxable = ""
            else:
                relaxable = "Relaxable"
            plot.edge(constraint.source.id, constraint.sink.id, label="{}: {}, {}".format(constraint.description, intervals, relaxable))
        for contingent in contingents:
            if contingent.type == "stcu":
                plot.edge(contingent.source.id, contingent.sink.id, label="{}: [{}, {}]".format(constraint.description, round(contingent.lb, dp), round(contingent.ub, dp)), color="blue", fontcolor="blue")
            else:
                if contingent.distribution["type"] == "gaussian":
                    plot.edge(contingent.source.id, contingent.sink.id, label="{}: N({}, {})".format(contingent.description, round(contingent.mu, dp), round(contingent.sigma, dp)), color="red", fontcolor="red")
                elif contingent.distribution["type"] == "uniform":
                    plot.edge(contingent.source.id, contingent.sink.id, label="{}: U({}, {})".format(contingent.description, round(contingent.dist_lb, dp), round(contingent.dist_ub, dp)), color="red", fontcolor="red")
        try:
            plot.render('{}_plot.pdf'.format(self.name), view=True)
        except subprocess.CalledProcessError:
            print("Please close the PDF and rerun the script")

    def printPSTN(self, write=False):
        if write == False:
            for constraint in self.getConstraints():
                print("\n")
                print("Description = ", constraint.description)
                print("Type = ", constraint.type)
                print("Source = ", constraint.source, constraint.source.id)
                print("Sink = ", constraint.sink, constraint.sink.id)
                print("Interval = ", constraint.intervals)
                if constraint.hard == False:
                    print("Relaxable = True")
                else:
                    print("Relaxable = False")
                if constraint.type == "pstc":
                    print("Distribution = ", constraint.distribution)
        else:
            f = open("{}_print.txt".format(self.name), "w")
            for constraint in self.getConstraints():
                f.write("\n")
                f.write("\nDescription = " + constraint.description)
                f.write("\nType = " + constraint.type)
                f.write("\nSource = " + constraint.source.id)
                f.write("\nSink = " + constraint.sink.id)
                f.write("\nInterval = " + str(constraint.intervals))
                if constraint.type == "pstc":
                    f.write("\nDistribution = " +  str(constraint.distribution))
                try:
                    f.write("\nIncoming Contingent at start = " + str(self.incomingContingent(constraint)["start"].description))
                    f.write("\nIncoming Contingent at end = " + str(self.incomingContingent(constraint)["end"]))
                except AttributeError:
                    f.write("\nIncoming Contingent at start = None")
                    f.write("\nIncoming Contingent at end = None")
            f.close()

    def returnTotalValue(self):
        val = 0
        for constraint in self.getConstraints():
            if constraint.type != "pstc":
                val += constraint.intervals["value"]
        return val
    
    def countType(self, _type):
        return len([i for i in self.getConstraints() if i.type == _type])
    
    def countUncontrollables(self):
        return len(self.getUncontrollableConstraints())
    
    def getProblemVariables(self):
        vars = [i.id for i in self.getControllables()]
        for constraint in self.constraints:
            if constraint.hard == False:
                vars.append(constraint.name + "_rl")
                vars.append(constraint.name + "_ru")
        return vars
        
    def getRandomVariables(self):
        rvars = []
        for constraint in self.constraints:
            if constraint.type == "pstc":
                name = "X" + "_" + constraint.source.id + "_" + constraint.sink.id
                rvars.append(name)
        return rvars


